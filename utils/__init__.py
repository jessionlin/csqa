import torch
import os
import csv
import json

import random
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
工具函数
1. mkdir_if_notexist(dir_)
2. get_device(gpu_ids)
3. _load_csv(file_name, skip_fisrt)
4. _load_json(file_name)
5. _save_json(data, file_name)
6. AvgVar()
7. Vn()
8. F1_Measure()
9. f1_measure(tp, fp, fn)
10. set_seed(args)
"""


def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_device(gpu_ids):
    if gpu_ids:
        device_name = 'cuda:' + str(gpu_ids[0])
        n_gpu = torch.cuda.device_count()
        print('device is cuda, # cuda is: %d' % n_gpu)
    else:
        device_name = 'cpu'
        print('device is cpu')
    device = torch.device(device_name)
    return device


def _load_csv(file_name, skip_first=True):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        if skip_first:  # 跳过表头
            f.__next__()
        for line in csv.reader(f):
            yield line


def _load_json(file_name):
    with open(file_name, encoding='utf-8', mode='r') as f:
        return json.load(f)


def _save_json(data, file_name):
    mkdir_if_notexist(file_name)
    with open(file_name, encoding='utf-8', mode='w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def _load_jsonl(file_name):
    lines = []
    with open(file_name, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            lines.append(json.loads(line))
            line = f.readline()
    return lines

class AvgVar:
    """
    维护一个累加求平均的变量
    """
    def __init__(self):
        self.var = 0
        self.steps = 0

    def inc(self, v, step=1):
        self.var += v
        self.steps += step

    def avg(self):
        return self.var / self.steps if self.steps else 0


class Vn:
    """
    维护n个累加求平均的变量
    """
    def __init__(self, n):
        self.n = n
        self.vs = [AvgVar() for i in range(n)]

    def __getitem__(self, key):
        return self.vs[key]

    def init(self):
        self.vs = [AvgVar() for i in range(self.n)]

    def inc(self, vs):
        for v, _v in zip(self.vs, vs):
            v.inc(_v)

    def avg(self):
        return [v.avg() for v in self.vs]

    def list(self):
        return [v.var for v in self.vs]


class F1_Measure:
    """
    ----------------
            真实
            P   N
    预   P  tp  fp
    测   N  fn  tn
    ----------------

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * prec * recall / (prec + recall)
       = 2 * tp / (tp + fp) * tp / (tp + fn) / [ tp / (tp + fp) + tp / (tp + fn)]
       = 2 * tp / [tp + fp + tp + fn]
    """
    def __init__(self):
        self.tp = 0
        self.tp_fp_tp_fn = 0

    def inc(self, tp, tp_fp, tp_fn):
        # tp_fp: 预测值为正的
        # tp_fn: 真实值为正的
        self.tp += tp
        self.tp_fp_tp_fn += tp_fp + tp_fn

    def f1(self):
        f1 = 2 * self.tp / self.tp_fp_tp_fn if self.tp else 0
        return f1


def f1_measure(tp, fp, fn):
    return 2 * tp / (tp + fp + tp + fn) if tp else 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
