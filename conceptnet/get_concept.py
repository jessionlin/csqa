import os
from common import load_jsonl, save_json, conceptnet_dir, root_dirs, csqa_data_dirs, mkdir_if_notexist
from rearrange_conceptnet import load_rearranged_conceptnet


target_relations = [
                    'CausesDesire',
                    'HasProperty',
                    'CapableOf',
                    'PartOf',
                    'AtLocation',
                    'Desires',
                    'HasPrerequisite',
                    'HasSubevent',
                    'Antonym',
                    'Causes'
                    ]

def enrich_word(word, conceptnet):
    try:
        edges = conceptnet[word.replace(' ', '_')]
    except:
        return []
    triples = []
    for edge in edges:
        if edge['rel'] not in target_relations:
            continue
        
        if edge['weight'] < 1 :
            continue

        start_splited = edge['start'].split('/')

if len(start_splited) >= 5:
    
    start = start_splited[3]
    
    end = edge['end'].split('/')[3]
        if len(end) < 1:
            continue
        
        else:
            start = start_splited[-1]
            end = edge['end'].split('/')[-1]

    try:
        triple = {
            'start': start.replace('_',' '),
            'rel': edge['rel'],
            'end': end.replace('_',' '),
            'surface_text':edge['surface_text'],
            'weight': edge['weight']
            }
            triples.append(triple)
        except:
            break

return triples


def triples_filter(triples):
    ret = []
    for rel in target_relations:
        for triple in triples:
            if triple['rel'] == rel:
                ret.append(triple)
        if len(ret) > 0:
            return ret
    return ret

def triples_score(triples):
    rel_dict = {}
    for triple in triples:
        try:
            rel_dict[triple['rel']].append(triple)
        except:
            rel_dict[triple['rel']] = [triple]
    scores_dict = {}
    for key in list(rel_dict.keys()):
        scores_dict[key] = 1.0 / len(rel_dict[key])
    
    filtered_rel = sorted(triples, key=lambda x: x['weight'] * scores_dict[x['rel']], reverse=True)
    return filtered_rel

def enrich_rel(datas, conceptnet, rel=None):
    cases = []
    index_i, index_j = 0, 0
    for i, data in enumerate(datas) :
        data['initial_id'] = i
        
        for choice in data['question']['choices']:
            triples = enrich_word(choice['text'], conceptnet)
            triples = triples_score(triples)
            triples_rel = []
            for triple in triples:
                if rel != None and triple['rel'] == rel:
                    triples_rel.append(triple)
                elif rel == None and triple['rel'] in target_relations:
                    triples_rel.append(triple)
            triples = triples_rel
            index_i += 1
            if not triples:
                choice['triple'] = []
                choice['surface'] = ''
                choice['weight'] = 0.0
                index_j += 1
            else:
                filtered_rel = triples_score(triples)
                try:
                    triple = filtered_rel[0]
                    choice['triple'] = [[triple['start'], triple['rel'], triple['end']]]
                    choice['surface'] = triple['surface_text']
                    choice['weight'] = triple['weight']
                except:
                    choice['triple'] = []
                    choice['surface'] = ''
                    choice['weight'] = 0.0
        # index_j += 1
    cases.append(data)
# 1348, 247
print("index is {}, {}".format(index_i, index_j))
    return cases


def enrich(datas, conceptnet):
    cases = []
    index_i, index_j = 0, 0
    for i, data in enumerate(datas) :
        data['initial_id'] = i
        triples = enrich_word(data['question']['question_concept'], conceptnet)
        objects = {triple['end']:triple for triple in triples}
        
        for choice in data['question']['choices']:
            try:
                '''
                    objects[choice['text']] = {
                    'start': start.replace('_',' '),
                    'rel': edge['rel'],
                    'end': end.replace('_',' '),
                    'surface_text':edge['surface_text'],
                    'weight' : edgge['weight']
                    }
                    '''
                triple = objects[choice['text']]
                choice['triple'] = [[triple['start'], triple['rel'], triple['end']]]
                choice['surface'] = triple['surface_text']
                choice['weight'] = triple['weight']
            except:
                triples = enrich_word(choice['text'], conceptnet)
                index_i += 1
                if not triples:
                    choice['triple'] = []
                    choice['surface'] = ''
                    choice['weight'] = 0.0
                    index_j += 1
                else:
                     rel_dict = {}
                     for triple in triples:
                         try:
                             rel_dict[triple['rel']].append(triple)
                         except:
                             rel_dict[triple['rel']] = [triple]
                     scores_dict = {}
                     for key in list(rel_dict.keys()):
                         scores_dict[key] = 1.0 / len(rel_dict[key])
                    
                    filtered_rel = sorted(triples, key=lambda x:x['weight'] * scores_dict[x['rel']], reverse=True)
                    print(filtered_rel)
                    try:
                        triple = filtered_rel[0]
                        choice['triple'] = [[triple['start'], triple['rel'], triple['end']]]
                        choice['surface'] = triple['surface_text']
                        choice['weight'] = triple['weight']
                    except:
                        choice['triple'] = []
                        choice['surface'] = ''
                        choice['weight'] = 0.0
                        index_j += 1
cases.append(data)
# 1348, 247
print("index is {}, {}".format(index_i, index_j))
    return cases


def check_entity(datas, conceptnet):
    index = 0
    whole = len(datas) * 6
    for data in datas:
        if enrich_word(data['question']['question_concept'], conceptnet):
            index += 1
        for choice in data['question']['choices']:
            if enrich_word(choice['text'], conceptnet):
                index += 1
    return float(index) / whole


if __name__ == '__main__':
    task = 'test'
    data_file = os.path.join(csqa_data_dirs, '{}_data.jsonl'.format(task))
    
    # rel = 'AtLocation'
    output_data_file_name = os.path.join(root_dirs, 'csqa_data', 'conceptnet','9_rels', '{}_data.json'.format(task))
    mkdir_if_notexist(output_data_file_name)
    # wnlemer = WordNetLemmatizer()
    
    datas = load_jsonl(data_file)
    print('loading conceptnet ...')
    conceptnet = load_rearranged_conceptnet(conceptnet_dir)
    print(f'conceptnet keys: {len(conceptnet)}')
    print('-----/n')
    
    
    cases = enrich(datas, conceptnet)
    print(len(cases))
    save_json(cases, output_data_file_name)

# dev_data
# AtLocation 没有找到三元组的比例：4489/7326
# Causes 6615/7326
# CapableOf 4976/7326
# Antonym 6347/7326
# HasSubevent 6466/7326
# HasPrerequisite 6403/7326
# CausesDesire 6492/7326
# Desires 7104/7326
# PartOf 5770/7326
# HasProperty 5355/7326

# train_data
# AtLocation 没有找到三元组的比例：36314/58446
# Causes 52872/58446
# CapableOf 40198/58446
# Antonym 50664/58446
# HasSubevent 51864/58446
# HasPrerequisite 51164/58446
# CausesDesire 52012/58446
# Desires 56624/58446
# PartOf 46800/58446
# HasProperty 43005/58446


# test_data
# AtLocation 没有找到三元组的比例：4208/6840
# Causes 6188/6840
# CapableOf 4637/6840
# Antonym 5947/6840
# HasSubevent 6026/6840
# HasPrerequisite 5979/6840
# CausesDesire 6063/6840
# Desires 6615/6840
# PartOf 5448/6840
# HasProperty 5018/6840

