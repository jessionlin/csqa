from .base_trainer import BaseTrainer
from .tensor import convert_to_tensor
from . import get_device

import torch
from torch import nn

from tqdm.autonotebook import tqdm
from transformers.modeling_albert import AlbertForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    if type(mlm_probability) is float:
        probability_matrix = torch.full(labels.shape, mlm_probability)
    else:
        probability_matrix = mlm_probability

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, print_step,
                 output_model_dir, tokenizer, fp16):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, vn=1)

        self.tokenizer = tokenizer
        self.fp16 = fp16
        print("fp16 is {}".format(fp16))

    def _forward(self, batch, record):
        if len(batch) == 0:
            inputs, labels = mask_tokens(batch[0], self.tokenizer)
        else:
            inputs, labels = mask_tokens(batch[0], self.tokenizer, batch[1])

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        loss = self.model(inputs, masked_lm_labels=labels)[0]
        loss = self._mean((loss,))[0]
        record.inc((loss.item(), ))
        return loss

    def _report(self, train_record):
        train_loss = train_record.avg()[0]
        print(f'_____Train loss {train_loss}')

    def train(self, epoch_num, train_dataloader):
        """
        去掉在验证集上的评估
        """
        self.global_step = 0
        self.train_record.init()

        for epoch in range(int(epoch_num)):
            print(f'---- Epoch: {epoch+1:02} ----')
            for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
                self.model.train()
                self._step(batch)

                if self.global_step % self.print_step == 0:

                    self._report(self.train_record)
                    self.train_record.init()

        self._report(self.train_record)
        self.save_model()

    def make_scheduler(self, optimizer, warmup_proportion, t_total):
        return get_linear_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)


class MaskedLM:
    """
    1. self.init()
    2. self.train(...)
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def init(self, ModelClass):
        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)

        print('init_model', self.config.bert_model_dir)
        model = ModelClass.from_pretrained(self.config.bert_model_dir)

        if multi_gpu:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        self.trainer = Trainer(
            model, multi_gpu, device,
            self.config.print_step,
            self.config.output_model_dir,
            self.tokenizer,
            self.config.fp16)

    def train(self, train_dataloader):
        t_total = len(train_dataloader) * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(self.config.num_train_epochs, train_dataloader)


def make_input_ids(tokens, tokenizer, max_seq_length, mask_probs=None):
    # tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens[:max_seq_length-2] + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [0] * (max_seq_length - len(input_ids))

    if mask_probs is not None:
        mask_probs = [0.] + mask_probs[:max_seq_length-2] + [0.] * (max_seq_length - len(mask_probs) - 1)
    if len(mask_probs) < max_seq_length:
        mask_probs = mask_probs + [0.] * (max_seq_length - len(mask_probs))
    
    # print("input_ids is {}, mask_probs is {}, max_seq_length is {}".format(len(input_ids), len(mask_probs), max_seq_length))
    assert len(input_ids) == max_seq_length
    assert len(mask_probs) == max_seq_length
    
    return input_ids, mask_probs


def make_dataloader(texts, tokenizer, batch_size, max_seq_length, mask_probs=None):
    """
    mask_probs: list or None
    """

    if mask_probs is not None:
        all_input_ids = []
        all_mask_probs = []

        for text, prob in zip(texts, mask_probs):
            input_ids, prob = make_input_ids(text, tokenizer, max_seq_length, prob)
            all_input_ids.append(input_ids)
            all_mask_probs.append(prob)

        data = (all_input_ids, all_mask_probs)

    else:

        all_input_ids = tuple(make_input_ids(text, tokenizer, max_seq_length)[0]
                              for text in texts)

        data = (all_input_ids, )
    # print("all_input_ids size is {}, {}".format(len(all_input_ids), len(all_input_ids[0])))

    return convert_to_tensor(data, batch_size, drop_last=False, shuffle=True)


def fine_tune_on_texts(texts, args, tokenizer, mask_probs=None):

    train_dataloader = make_dataloader(
        texts, tokenizer, batch_size=args.batch_size,
        max_seq_length=args.max_seq_length, mask_probs=mask_probs)

    mlm = MaskedLM(args, tokenizer)
    mlm.init(AlbertForMaskedLM)
    mlm.train(train_dataloader)
