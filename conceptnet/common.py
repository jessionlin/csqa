import os
import json

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dirs = r'/Users/linjiaxuan/PycharmProjects/neo4j/'
csqa_data_dirs = r'/Users/linjiaxuan/Documents/nlpData/commonsenseQA'
root_uri = "bolt://localhost:7687"
conceptnet_dir = r'/Users/linjiaxuan/Documents/nlpData/conceptnet_english'
# root_uri = "https://localhost:7687"

def get_file_list(file_path=None):
    if file_path is None:
        return False
    files = []
    for root, dir, file in os.walk(file_path):
        if file:
            for item in file:
                files.append(os.path.join(root, item))
    return files


def read_to_array(file_name,encoding='utf-8', cols=-1,sep=','):
    data = []
    with open(file_name, 'r', encoding=encoding) as f:
        contents = f.readlines()
        for content in contents:
            temp = content.replace('\n', '').split(sep)
            if cols == 2:
                temp = [temp[0], sep.join(temp[1:])]
            elif cols == 1:
                temp = ''.join(temp)
            data.append(temp)
    return data

def read_to_dict(filename, index = 0):
    if index > 1:
        return {}
    data = read_to_array(filename)
    ret = {}
    for dat in data:
        ret[dat[index]] = dat[1-index]
    return ret


def save_file(data, file_name, col=2, sep=',', title = []):
    if col == 1:
        content = [''.join(dat) for dat in data]
    else:
        content = [sep.join(dat) for dat in data]
    ps = '\n'.join(content)
    if len(title) == len(data[0]):
        ps = sep.join(title) + '\n' + ps
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(ps)

def norm_str(s):
    return s.replace('"', '').replace(',', ' ')

def load_json(file_name):
    text = None
    with open(file_name, 'r') as f:
        text = json.loads(f.read())
    # print(text)
    return text

def get_label(id):
    if id[0] == 'Q':
        if len(id) > 6:
            label = id[:-5]
        else:
            label = 'Q0'
    else:
        label = 'bad'
    return label

def save_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def get_mapping(file):
    items = read_to_array(file, sep='\t')
    items_dict = {}
    for item in items:
        items_dict[item[0]] = item[1]
    return items_dict


def load_jsonl(file):
    questions = []
    with open(file, 'r') as f:
        content = f.readline()
        while len(content) > 0:
            temp = json.loads(content)
            questions.append(temp)
            content = f.readline()
    return questions


def make_file(file_name):
    print(file_name)
    if file_name[-1] == '/' or '.' not in file_name.split('/')[-1]:
        file_name = os.path.join(file_name, 't.txt')
        print(file_name)
    if not os.path.exists(file_name):
        if not os.path.isdir(file_name):
            (path, file) = os.path.split(file_name)
            if not os.path.exists(path):
                os.makedirs(path)
            try:
                fp = open(file_name, 'w')
                fp.close()
            except:
                pass

def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# def replace_prep(content, choice):
#     # index = 0
#     # for prep in prep_list:
#     #     if prep in content:
#     #         index = 1
#     #         content = content.replace(prep, ' {} '.format(choice))
#     #         break
#     # if index == 0:
#     #     content = content.strip(' ')
#     #     if content[-1] == '.':
#     #         content = content[:-1]
#     #     content += ' {}'.format(choice)
#     # return content
#     for prep in prep_list:
#         if prep in content:
#             temp = content.split(' ')
#             if temp[-1] == '{}?'.format(prep) or (temp[-1] == '?' and temp[-2] == prep):
#                 content = content.replace(prep, ' {} '.format(choice))
#     return content
recorrect_dict = {
    'ca': 'can',
    'n\'t': 'not',
    '\'ve': 'have'
}


def _recorrect(word):
    return word if word not in recorrect_dict else recorrect_dict[word]

def _pos(pos):
    """还原词形"""
    if pos.startswith('NN'):
        return 'n'

    elif pos.startswith('VB'):
        return 'v'

    elif pos.startswith('JJ'):
        return 'a'

    elif pos.startswith('R'):
        return 'r'

    else:
        return 'unknown'


def _lemmatize(word, pos, wnlem):
    """还原词形"""
    if pos.startswith('NN'):
        word = wnlem.lemmatize(word, pos='n')

    elif pos.startswith('VB'):
        word = wnlem.lemmatize(word, pos='v')

    elif pos.startswith('JJ'):
        word = wnlem.lemmatize(word, pos='a')

    elif pos.startswith('R'):
        word = wnlem.lemmatize(word, pos='r')

    return _recorrect(word)
