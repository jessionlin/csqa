import json
from collections import defaultdict
import os


def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def _save_json(data, file_name):
    mkdir_if_notexist(file_name)
    with open(file_name, encoding='utf-8', mode='w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _load_json(file_name):
    # print(file_name)
    with open(file_name, encoding='utf-8', mode='r') as f:
        return json.load(f)


class Edge:
    def __init__(self, start, end, rel, json_obj):
        self.start = start
        self.end = end
        self.rel = rel
        self.weight = json_obj['weight']
        self.surface_text = '' if 'surfaceText' not in json_obj else json_obj['surfaceText']

    def to_json(self):
        return {
            'start': self.start,
            'end': self.end,
            'rel': self.rel,
            'weight': self.weight,
            'surface_text': self.surface_text
        }


def _load(file_name):
    with open(file_name, encoding='gbk', mode='r') as f:
        count = 0
        for line in f:
            count = count + 1
            if (count+1) % 40000 == 0:
                print('##', count+1)

            line = line.strip().split('\t')
            _, rel, start, end, json_str = line
            json_obj = json.loads(json_str)

            # if 'surfaceText' not in json_obj:
            #     continue

            if json_obj['weight'] < 1:
                continue

            rel = rel.split('/')[-1]
            if rel in ['RelatedTo', 'ExternalURL']:
                continue

            if '/en/' not in start or '/en/' not in end:
                continue

            edge = Edge(start, end, rel, json_obj)

            yield edge


"""
英语
置信度>1
关系筛选
"""


def load_rearranged_conceptnet(dirname):
    conceptnet = {}
    for filename in os.listdir(dirname):
        pathname = os.path.join(dirname, filename)
        if pathname[-9:] == '.DS_Store':
            continue
        _conceptnet = _load_json(pathname)
        conceptnet.update(_conceptnet)
    return conceptnet


if __name__ == '__main__':
    conceptnet = load_rearranged_conceptnet('/Users/linjiaxuan/Documents/nlpData/conceptnet_english')
    keys = list(conceptnet.keys())
    print("keys length is {}".format(len(keys)))

    print(conceptnet)
