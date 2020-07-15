'''
作为baseline，输入格式为initial_question [SEP] question_concept [SEP] choice_concept
accuracy:0.8043
batch_size:8
step:5

without apex
accuracy:0.8190
batch_size:8
step:5
'''

'''
使用ernie进行测试，输入格式为initial_question [SEP] question_triple [SEP] choice_triple ，倘若某个concept没有选定的triple，则使用其concept，triple选择策略为先通过word2vec根据des字段选择某一特定实体，然后在其中通过ERNIE进行选择合适的三元组
accuracy:0.7985
batch_size:8
step:5
'''

'''
使用ernie进行测试，输入格式为initial_question [SEP] question_triple [SEP] choice_triple ，倘若某个concept没有选定的triple，则使用其concept，triple选择策略，通过ERNIE进行选择合适的三元组（所有相关实体的三元组）
accuracy:0.7944
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] question_triple [SEP] choice_triple，倘若某个concept没有选定的triple，则使用其concept，triple选择策略为通过concept获取所有相关实体，
获取到所有实体的常识三元组，并将三元组映射为 subject relation object格式，通过Word2vec计算三元组与原始问题的相似度，选择相似度最高的一个三元组作为使用的常识三元组
accuracy:0.8026
choice_triple_num: 1
batch_size:8
step:5

accuracy:0.7903
choice_triple_num: 2
batch_size:8
step:5

accuracy:0.7895
choice_triple_num: 3
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] question_descriptions [SEP] choice_descriptions
使用导入了Wikidata20200418的neo4j图数据，找到由question_concept到choice_concept的3 hop以内的路径，获取两端节点，即question_entity和choice_entity，获取其descriptions，
如果没有找到相关路径，则将其concept赋值给对应的descriptions
accuracy:0.7854
batch_size:8
max_seq_length: 64
step:5
'''

'''
输入格式为initial_question [SEP] temp 
使用导入了Wikidata20200418的neo4j图数据，找到由question_concept到choice_concept的3 hop以内的路径，获取路径内所有三元组，如果某个choice能够得到这样的三元组路径，则temp就是这些路径上三元组的组合，temp = triple[0] [SEP] triple[1] [SEP] triple[2]，
如果没有，则temp = question_concept [SEP] choice_concept
accuracy:0.8018
batch_size:8
max_seq_length: 64
step:5

without apex
accuracy:0.7920
batch_size:8
max_seq_length: 64
step:5
'''

'''
输入格式为initial_question [SEP] question_concept [SEP] choice_concept [SEP] triple[0] [SEP] triple[1] [SEP] triple[2]
without apex
accuracy:0.7993
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] question_concept [SEP] choice_concept [SEP] triple
其中triple通过conceptnet获取，subject为question_concept，object为choice_concept， 若没有对应的triple，则去掉最后面的[SEP] triple
模型为conceptnet/initial
without apex
accuracy:0.8280
batch_size:8
step:5
'''


'''
输入格式为initial_question [SEP] temp
当含有subject为question_concept，object为choice_concept的三元组时，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept
模型为conceptnet/initial_2
without apex
accuracy:0.8280
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有subject为question_concept，object为choice_concept的三元组时，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept
其中对关系的类型进行限制，只选择论文中统计的关系类型，约占比例96%
模型为conceptnet/initial_3
without apex
accuracy:0.8296
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有subject为question_concept，object为choice_concept的三元组时，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept
其中，如果没有subject为question_concept，object为choice_concept的三元组时，会选择一个subject为choice_concept的三元组，选择weight最大的那一个
模型为conceptnet/fillna
without apex
accuracy:0.8206
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有subject为question_concept，object为choice_concept的三元组时，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept
其中，如果没有subject为question_concept，object为choice_concept的三元组时，会选择一个subject为choice_concept的三元组，
三元组选择策略，对每一个concept获取到的三元组，对其获取到的三元组进行关系权重评分，权重为其数量的倒数，最后按照三元组的weight与关系的权重乘积作为最终的权重得分，选择得分最高的一个三元组
模型为conceptnet/weight_rel
without apex
accuracy:0.8329
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有subject为question_concept，object为choice_concept的三元组时，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept
其中，如果没有subject为question_concept，object为choice_concept的三元组时，会选择一个subject为choice_concept的三元组，
三元组选择策略，对每一个concept获取到的三元组，对其获取到的三元组进行关系权重评分，权重为其数量的倒数，最后按照三元组的weight、关系的权重、以及不同rel在数据集中比例(按照论文中所述)乘积作为最终的权重得分，选择得分最高的一个三元组
模型为conceptnet/weight_rel_key
without apex
accuracy:0.8296
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有surface时，temp = surface.replace('[','').replace(']','')
否则，temp = question_concept [SEP] choice_concept
模型为conceptnet/descriptions
without apex
accuracy:0.8092
batch_size:8
step:5
'''

'''
输入格式为initial_question [SEP] temp
当含有对choice_concept扩充三元组，选择不同的relations，temp = subject relations object
否则，temp = question_concept [SEP] choice_concept

模型为conceptnet/XX（relations）
without apex
batch_size:8
step:5
accuracy:
Antonym:0.7936
AtLocation:0.7993
CapableOf:0.8010
Causes:0.7920
CausesDesire:0.8067
Desires:0.7977
HasPrerequisite:0.7969
HasProperty:0.8059
HasSubevent:0.7764
PartOf:0.8018

4rels:0.8010
9rels:
filter:0.8010
'''