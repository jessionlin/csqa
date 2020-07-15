from utils.base_trainer import BaseTrainer
from utils import get_device

import argparse
import torch
from torch import nn
import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    print("apex not imported")


class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, print_step,
                 output_model_dir, fp16):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, vn=3)
        self.fp16 = fp16
        self.multi_gpu = multi_gpu
        print("fp16 is {}".format(fp16))
            
    def clip_batch(self, batch):
        """
        设batch中最长句子的长度为max_seq_length, 将超过max_seq_length的部分删除
        """
        # print("batch size is {}".format(len(batch[0])))
        idx, input_ids, attention_mask, token_type_ids, labels = batch
        # [batch_size, 2, L]
        batch_size = input_ids.size(0)
        while True:
            end_flag = False
            for i in range(batch_size):
                if input_ids[i, 0, -1] != 0:
                    end_flag = True
                if input_ids[i, 1, -1] != 0:
                    end_flag = True 
            
            if end_flag:
                break
            else:
                input_ids = input_ids[:, :, :-1]
        
        max_seq_length = input_ids.size(2)
        attention_mask = attention_mask[:, :, :max_seq_length]
        token_type_ids = token_type_ids[:, :, :max_seq_length]
        
        return idx, input_ids, attention_mask, token_type_ids, labels
        
    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1) 
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1
        
    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            
            self.model = model
        self.optimizer = optimizer

    def _forward(self, batch, record):
        batch = self.clip_batch(batch)
        batch = tuple(t.to(self.device) for t in batch)
        result = self.model(*batch)
        result = self._mean(result)
        record.inc([it.item() for it in result])
        return result[0]

    def _report(self, train_record, devlp_record):
        # record: loss, right_num, all_num
        train_loss = train_record[0].avg()
        devlp_loss = devlp_record[0].avg()

        trn, tan = train_record.list()[1:]
        drn, dan = devlp_record.list()[1:]

        logger.info(f'\n____Train: loss {train_loss:.4f} {int(trn)}/{int(tan)} = {int(trn)/int(tan):.4f} |'
              f' Devlp: loss {devlp_loss:.4f} {int(drn)}/{int(dan)} = {int(drn)/int(dan):.4f}')


class SelectReasonableText:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, config):
        self.config = config

    def init(self, ModelClass):
        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
#        multi_gpu = gpu_ids
        device = get_device(gpu_ids)

        print('init_model', self.config.bert_model_dir)
        model = ModelClass.from_pretrained(self.config.bert_model_dir)
        print(model)

        if multi_gpu:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        self.trainer = Trainer(
            model, multi_gpu, device,
            self.config.print_step, self.config.output_model_dir, self.config.fp16)

    def train(self, train_dataloader, devlp_dataloader, save_last=True):
        t_total = len(train_dataloader) * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, train_dataloader, devlp_dataloader, save_last=save_last)

    @classmethod
    def load(cls, config, ConfigClass, ModelClass):
        gpu_ids = list(map(int, config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)

        srt = cls(config)
        srt.trainer = Trainer.load_model(
            ConfigClass, ModelClass, multi_gpu, device,
            config.print_step, config.output_model_dir, config.fp16)

        return srt

    def trial(self, dataloader, desc='Eval'):
        result = []

        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                ret = self.model.predict(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda())
                ret_max = torch.max(ret,1)[1]
                result.extend(ret_max.cpu().numpy().tolist())
            
        print("result length is {}, first 30 is {}".format(len(result), result[:30]))
        return result


def get_args():
    parser = argparse.ArgumentParser()

    # 训练过程中的参数
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # 路径参数
    parser.add_argument('--train_file_name', type=str)
    parser.add_argument('--devlp_file_name', type=str)
    parser.add_argument('--trial_file_name', type=str)
    parser.add_argument('--pred_file_name', type=str)
    parser.add_argument('--output_model_dir', type=str)
    parser.add_argument('--bert_model_dir', type=str)
    parser.add_argument('--bert_vocab_dir', type=str)

    # 其他参数
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--gpu_ids', type=str, default='0', help='以空格分割')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--fp16', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import time
    start = time.time()
    print("start is {}".format(start))
    import random
    import numpy as np

    from transformers.tokenization_albert import AlbertTokenizer
    from transformers.modeling_albert import AlbertConfig

    from specific.io import load_data
    from specific.tensor import make_dataloader
    from model.model import Model
    from utils.common import mkdir_if_notexist

    args = get_args()
    args.fp16 = True if args.fp16 == 1 else False
    
    
    print("args.fp16 is {}".format(args.fp16))
    assert args.mission in ('train', 'output')

    # ------------------------------------------------#
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # ------------------------------------------------#

    # ------------------------------------------------#
    experiment = 'conceptnet'
    print('load_data', args.train_file_name)
    train_data = load_data(experiment, args.train_file_name, type='json')

    print('load_data', args.devlp_file_name)
    devlp_data = load_data(experiment, args.devlp_file_name, type='json')
    
    print('get dir {}'.format(args.output_model_dir))
    mkdir_if_notexist(args.output_model_dir)
    print('load_vocab', args.bert_vocab_dir)
    tokenizer = AlbertTokenizer.from_pretrained(args.bert_vocab_dir)
    # ------------------------------------------------#

    # ------------------------------------------------#
    # print('make dataloader ...')
    if args.mission == 'train':
        train_dataloader = make_dataloader(
            experiment, train_data, tokenizer, batch_size=args.batch_size,
            drop_last=False, max_seq_length=64)  # 52 + 3

        print('train_data %d ' % len(train_data))

    devlp_dataloader = make_dataloader(
            experiment, devlp_data, tokenizer, batch_size=args.batch_size,
            drop_last=False, max_seq_length=64)

    print('devlp_data %d ' % len(devlp_data))
    # ------------------------------------------------#

    # -------------------- main ----------------------#
    if args.mission == 'train':
        srt = SelectReasonableText(args)
        srt.init(Model)
        srt.train(train_dataloader, devlp_dataloader, save_last=False)

        srt = SelectReasonableText
    elif args.mission == 'output':
        srt = SelectReasonableText(args)
        srt.load(args, AlbertConfig, Model)
        raise NotImplementedError
        srt.output_result(devlp_dataloader, args.pred_file_name)
    # ------------------------------------------------#
    
    end = time.time()
    logger.info("start is {}, end is {}".format(start, end))
    logger.info("循环运行时间:%.2f秒"%(end-start))
    with open('./result_1.txt', 'w', encoding='utf-8') as f:
        f.write("循环运行时间:%.2f秒"%(end-start))
