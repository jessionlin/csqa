import os

import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_csv(data, path,sep=',', type='default'):
    if len(data) < 1:
        return 
    content = ''
    if type == 'default':
        for dat in data:
            temp = sep.join(dat) + '\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(temp)
            
            
def make_file(fileName):
    if fileName[-1] == '/':
        fileName = fileName + 'mk.txt'
    if not os.path.exists(fileName):
        if not os.path.isdir(fileName):
            (path, file) = os.path.split(fileName)
            if not os.path.exists(path):
                os.makedirs(path)
            try:
                fp = open(fileName, 'w')
                fp.close()
            except:
                pass


def mkdir_if_notexist(dir_):
    dirname, filename = os.path.split(dir_)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
            



