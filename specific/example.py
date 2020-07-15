"""
TaskAExample(...)
TaskBExample(...)
TaskCExample() # 尚未完成
"""
from utils.feature import Feature, KBERTFeature
from utils.edit_distance import diff_seq, diff_seq_three

label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

class Example:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['question']['choices']
        question_concept = json_obj['question']['question_concept']
        # text1 = ' {} [SEP] {} [SEP] {} '.format(choices[0]['filter'], question_concept, choices[0]['text'])
        # text2 = ' {} [SEP] {} [SEP] {} '.format(choices[1]['filter'], question_concept, choices[1]['text'])
        # text3 = ' {} [SEP] {} [SEP] {} '.format(choices[2]['filter'], question_concept, choices[2]['text'])
        # text4 = ' {} [SEP] {} [SEP] {} '.format(choices[3]['filter'], question_concept, choices[3]['text'])
        # text5 = ' {} [SEP] {} [SEP] {} '.format(choices[4]['filter'], question_concept, choices[4]['text'])
        
        # question_concept = json_obj['questions']['text']
        text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['question']['stem'], question_concept, choices[0]['text'])
        text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['question']['stem'], question_concept, choices[1]['text'])
        text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['question']['stem'], question_concept, choices[2]['text'])
        text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['question']['stem'], question_concept, choices[3]['text'])
        text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['question']['stem'], question_concept, choices[4]['text'])
        
        try:
            label = label_dict[json_obj['answerKey']]
        except:
            label = -1
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label,
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }

class MulClassExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)


    def get_input_choice(self, question,question_concept, choice, type='1'):
        length = len(choice['text'].split(' '))
        if length == 1 or length == 3:
            if len(choice['descriptions']) > 0:
                text = ' {} [SEP] {} [SEP] {}'.format(question, choice['text'], choice['descriptions'])
            else:
                text = ' {} [SEP] {} [SEP] {}'.format(question, question_concept, choice['text'])
        else:
            text = ' {} [SEP] {} [SEP] {}'.format(question, question_concept, choice['text'])
        return text

        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        
        question_concept = json_obj['questions']['text']
        text1 = cls.get_input_choice(cls, json_obj['content'], question_concept, choices[0])
        text2 = cls.get_input_choice(cls, json_obj['content'], question_concept, choices[1])
        text3 = cls.get_input_choice(cls, json_obj['content'], question_concept, choices[2])
        text4 = cls.get_input_choice(cls, json_obj['content'], question_concept, choices[3])
        text5 = cls.get_input_choice(cls, json_obj['content'], question_concept, choices[4])
        # text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],question_concept,  choices[0]['text'])
        # if len(choices[0]['text'].split(' ')) == 1:
        #     text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],choices[0]['text'] , choices[0]['descriptions'])

        # text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],question_concept,  choices[1]['text'])
        # if len(choices[1]['text'].split(' ')) == 1:
        #     text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],choices[1]['text'] , choices[1]['descriptions'])

        # text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],question_concept,  choices[2]['text'])
        # if len(choices[2]['text'].split(' ')) == 1:
        #     text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],choices[2]['text'] , choices[2]['descriptions'])

        # text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],question_concept,  choices[3]['text'])
        # if len(choices[3]['text'].split(' ')) == 1:
        #     text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],choices[3]['text'] , choices[3]['descriptions'])

        # text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],question_concept,  choices[4]['text'])
        # if len(choices[4]['text'].split(' ')) == 1:
        #     text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'],choices[4]['text'] , choices[4]['descriptions'])
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }

class DesExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        question_concept = json_obj['questions']['text']
        if len(json_obj['questions']['descriptions']) > 0:
            question_des = json_obj['questions']['descriptions']
        else:
            question_des = json_obj['questions']['text']

        choice_des = choices[0]['text']
        if len(choices[0]['descriptions']) > 0:
            choice_des = choices[0]['descriptions']
        text1 = ' {} [SEP] {} [SEP] {}'.format(json_obj['content'],  choices[0]['text'], choice_des)

        choice_des = choices[1]['text']
        if len(choices[1]['descriptions']) > 0:
            choice_des = choices[1]['descriptions']
        text2 = ' {} [SEP] {} [SEP] {}'.format(json_obj['content'],  choices[1]['text'], choice_des)

        choice_des = choices[2]['text']
        if len(choices[2]['descriptions']) > 0:
            choice_des = choices[2]['descriptions']
        text3 = ' {} [SEP] {} [SEP] {}'.format(json_obj['content'],  choices[2]['text'], choice_des)

        choice_des = choices[3]['text']
        if len(choices[3]['descriptions']) > 0:
            choice_des = choices[3]['descriptions']
        text4 = ' {} [SEP] {} [SEP] {}'.format(json_obj['content'],  choices[3]['text'], choice_des)

        choice_des = choices[4]['text']
        if len(choices[4]['descriptions']) > 0:
            choice_des = choices[4]['descriptions']
        text5 = ' {} [SEP] {} [SEP] {}'.format(json_obj['content'],  choices[4]['text'], choice_des)
        
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }
class WikiKProExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(choice1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        question_concept = json_obj['questions']['text']
        key = 'target_triple'
        try:
            if len(json_obj['questions'][key]) > 0:
                question_concept =' '.join(json_obj['questions'][key])
        except:
            pass
        choice_concept = choices[0]['text']
        try:
            if len(choices[0][key]) > 0:
                choice_concept = ' '.join(choices[0][key])
        except:
            pass
        text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        # print(text1)
        
        choice_concept = choices[1]['text']
        try:
            if len(choices[1][key]) > 0:
                choice_concept = ' '.join(choices[1][key])
        except:
            pass
        text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[2]['text']
        try:
            if len(choices[2][key]) > 0:
                choice_concept = ' '.join(choices[2][key])
        except:
            pass
        text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[3]['text']
        try:
            if len(choices[3][key]) > 0:
                choice_concept = ' '.join(choices[3][key])
        except:
            pass
        text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[4]['text']
        try:
            if len(choices[4][key]) > 0:
                choice_concept = ' '.join(choices[4][key])
        except:
            pass
        text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        
        
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }


class MulTriplesExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(choice1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
    
    def choice_triple_content(self, triples):
        temps = [' '.join(triple) for triple in triples]
        content = ' '.join(temps)
        return content
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        question_concept = json_obj['questions']['text']
        key = 'target_triple'
        try:
            if len(json_obj['questions'][key]) > 0:
                question_concept =' '.join(json_obj['questions'][key])
        except:
            pass
        choice_concept = choices[0]['text']
        try:
            if len(choices[0][key]) > 0:
                temps = ['  '.join(triple) for triple in choices[0][key]]
                choice_concept = temps[0]
        except:
            pass
        text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        choice_concept = choices[1]['text']
        try:
            if len(choices[1][key]) > 0:
                temps = [' '.join(triple) for triple in choices[1][key]]
                choice_concept = temps[0]
        except:
            pass
        text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[2]['text']
        try:
            if len(choices[2][key]) > 0:
                temps = [' '.join(triple) for triple in choices[2][key]]
                choice_concept = temps[0]
        except:
            pass
        text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[3]['text']
        try:
            if len(choices[3][key]) > 0:
                temps = [' '.join(triple) for triple in choices[3][key]]
                choice_concept = temps[0]
        except:
            pass
        text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[4]['text']
        try:
            if len(choices[4][key]) > 0:
                temps = [' '.join(triple) for triple in choices[4][key]]
                choice_concept = temps[0]
        except:
            pass
        text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        
        
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }
        

class WikiExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        question_concept = json_obj['questions']['text']


        if len(json_obj['questions']['knowledge']) > 0:
            question_concept = ' '.join(json_obj['questions']['knowledge'][0])
        choice_concept = choices[0]['text']
        if len(choices[0]['knowledge']) > 0:
            choice_concept = ' '.join(choices[0]['knowledge'][0])
        text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[1]['text']
        if len(choices[1]['knowledge']) > 0:
            choice_concept = ' '.join(choices[1]['knowledge'][0])
        text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[2]['text']
        if len(choices[2]['knowledge']) > 0:
            choice_concept = ' '.join(choices[2]['knowledge'][0])
        text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[3]['text']
        if len(choices[3]['knowledge']) > 0:
            choice_concept = ' '.join(choices[3]['knowledge'][0])
        text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        
        choice_concept = choices[4]['text']
        if len(choices[4]['knowledge']) > 0:
            choice_concept = ' '.join(choices[4]['knowledge'][0])
        text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choice_concept)
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }
    
class WikiKBERTExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(self.text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        return cls(
            json_obj['id'],
            choices[0]['kbert'],
            choices[1]['kbert'],
            choices[2]['kbert'],
            choices[3]['kbert'],
            choices[4]['kbert'],
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Text3': self.text3,
            'Text4': self.text4,
            'Text5': self.text5,
            'Label': self.label
        }
    

class Neo4jExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['choice']
        question_concept = json_obj['questions']['text']
        # question_concept = json_obj['questions']['descriptions']
        def mkinput(question_concept, choice):
            triples_temp = question_concept + ' [SEP] ' + choice['text'] 
            if choice['triples'] != choice['text']:
                triples = [' '.join(triple) for triple in choice['triples']]
                triples = ' [SEP] '.join(triples)
                triples_temp = triples_temp + ' [SEP] ' + triples
                # print(triples_temp)
            text = ' {} [SEP] {} '.format(json_obj['content'], triples_temp)
            return text

        text1 = mkinput(question_concept, choices[0])
        text2 = mkinput(question_concept, choices[1])
        text3 = mkinput(question_concept, choices[2])
        text4 = mkinput(question_concept, choices[3])
        text5 = mkinput(question_concept, choices[4])
        # text1 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[0]['descriptions'])
        # text2 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[1]['descriptions'])
        # text3 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[2]['descriptions'])
        # text4 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[3]['descriptions'])
        # text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[4]['descriptions'])
         
        return cls(
            json_obj['id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label_dict[json_obj['answerKey']],
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }

class ConceptNetExample:
    def __init__(self, idx, choice1, choice2, choice3, choice4, choice5, label = -1):
        self.idx = idx
        self.text1 = choice1
        self.text2 = choice2
        self.text3 = choice3
        self.text4 = choice4
        self.text5 = choice5
        self.label = int(label)
        # print("text1 is {}".format(text1))
    
    def __str__(self):
        return f"{self.idx} | {self.text1} | {self.text2} | {self.text3} | {self.text4} | {self.text5} | {self.label}"
        
    def fl(self, tokenizer, max_seq_length):
        fs = self.f(tokenizer, max_seq_length)
        return (*fs, self.label)
        
    def f(self, tokenizer, max_seq_length):
        tokens1 = tokenizer.tokenize(self.text1)
        tokens2 = tokenizer.tokenize(self.text2)
        tokens3 = tokenizer.tokenize(self.text3)
        tokens4 = tokenizer.tokenize(self.text4)
        tokens5 = tokenizer.tokenize(self.text5)

        feature1 = Feature.make_single(self.idx, tokens1, tokenizer, max_seq_length)
        feature2 = Feature.make_single(self.idx, tokens2, tokenizer, max_seq_length)
        feature3 = Feature.make_single(self.idx, tokens3, tokenizer, max_seq_length)
        feature4 = Feature.make_single(self.idx, tokens4, tokenizer, max_seq_length)
        feature5 = Feature.make_single(self.idx, tokens5, tokenizer, max_seq_length)
        # feature = Feature.make(self.idx, tokens1, tokens2, tokenizer, max_seq_length)
        # print(feature1.input_mask)
        return (feature1, feature2, feature3, feature4, feature5)
        
        
    @classmethod
    def load_from_json(cls, json_obj):
        choices = json_obj['question']['choices']
        question_concept = json_obj['question']['question_concept']
        def mkinput(question_concept, choice):
            # triples_temp = question_concept + ' [SEP] ' + choice['text'] 
            if choice['triple']:
                # triples = choice['triple']['start'] + ' ' + choice['triple']['rel'] + ' ' + choice['triple']['end']
                triples = ' '.join(choice['triple'][0])
                # triples = choice['surface'].replace('[','').replace(']','')
                # triples_temp = triples_temp + ' [SEP] ' + triples
                triples_temp = triples
            else:
                # triples_temp = triples_temp + ' [SEP] ' + choice['text']
                triples_temp = question_concept + ' [SEP] ' + choice['text']
            text = ' {} [SEP] {} '.format(json_obj['question']['stem'], triples_temp)
            # print(text)
            return text

        text1 = mkinput(question_concept, choices[0])
        text2 = mkinput(question_concept, choices[1])
        text3 = mkinput(question_concept, choices[2])
        text4 = mkinput(question_concept, choices[3])
        text5 = mkinput(question_concept, choices[4])
         # text5 = ' {} [SEP] {} [SEP] {} '.format(json_obj['content'], question_concept, choices[4]['descriptions'])
        try:
            label =  label_dict[json_obj['answerKey']]
        except:
            label = -1
        return cls(
            json_obj['initial_id'],
            text1,
            text2,
            text3,
            text4,
            text5,
            label,
        )

    def to_json(self):
        return {
            'ID': self.idx,
            'Text1': self.text1,
            'Text2': self.text2,
            'Label': self.label
        }