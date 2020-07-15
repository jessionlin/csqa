'''
生成验证test数据集结果的CSV文件
'''
import os
from common import root_dirs, load_json, read_to_array, save_file

label_dict = {
    '0':'A',
    '1':'B',
    '2':'C',
    '3':'D',
    '4':'E',
}
if __name__ == '__main__':
    results_file = os.path.join(root_dirs, 'results', 'weight_rel.csv')
    gold_file = os.path.join(root_dirs, 'csqa_data', 'conceptnet', 'weight_rel', 'test_data.json')

    results = read_to_array(results_file)
    golds = load_json(gold_file)
    print(results[:2])
    print(golds[:2])
    assert  len(results) == len(golds)
    length = len(results)
    print(results[0])
    print(golds[0])

    tests = []
    for i in range(length):
        assert int(results[i][0]) == golds[i]['initial_id']
        tests.append([golds[i]['id'], label_dict[results[i][-1]]])
    print(tests[0])
    print(len(tests))
    save_file(tests, os.path.join(root_dirs, 'test_results', 'predictions.csv'))

