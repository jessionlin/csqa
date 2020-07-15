from utils import _load_json, _save_json, _load_csv
from .example import  WikiExample,MulTriplesExample, WikiKBERTExample, Example, WikiKProExample, DesExample, MulClassExample,ConceptNetExample, Neo4jExample


def load_data(task, *args, **kwargs):
    assert task in ('exp','exp_0', 'exp_1', 'exp_2', 'exp_3', 'exp_4', 'exp_5', 'neo4j', 'conceptnet')

    if task == 'exp':
        return _load_data(*args, **kwargs)
    elif task == 'exp_0':
        return _load_data_0(*args, **kwargs)
    elif task == 'exp_1':
        return _load_data_1(*args, **kwargs)
    elif task == 'exp_2':
        return _load_data_2(*args, **kwargs)
    elif task == 'exp_3':
        return _load_data_3(*args, **kwargs)
    elif task == 'exp_4':
        return _load_data_4(*args, **kwargs)
    elif task == 'exp_5':
        return _load_data_5(*args, **kwargs)
    elif task == 'neo4j':
        return _load_data_neo4j(*args, **kwargs)
    elif task == 'conceptnet':
        return _load_data_conceptnet(*args, **kwargs)

    


def save_data(examples, file_name):
    data = []
    for example in examples:
        data.append(example.to_json())
    _save_json(data, file_name)
    
def _load_data(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = Example.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_0(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = WikiExample.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_1(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = WikiKBERTExample.load_from_json(json_obj)
            examples.append(example)
    return examples
    
def _load_data_2(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = WikiKProExample.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_3(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = DesExample.load_from_json(json_obj)
            examples.append(example)
    return examples
    
def _load_data_4(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = MulClassExample.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_5(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = MulTriplesExample.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_neo4j(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = Neo4jExample.load_from_json(json_obj)
            examples.append(example)
    return examples

def _load_data_conceptnet(file_name, type='json'):
    examples = []
    if type == 'json':
        for json_obj in _load_json(file_name):
            example = ConceptNetExample.load_from_json(json_obj)
            examples.append(example)
    return examples