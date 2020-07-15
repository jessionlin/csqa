from .layers import AttentionMerge

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers.modeling_albert import AlbertPreTrainedModel
from transformers.modeling_albert import AlbertModel


class KBERT(AlbertModel):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        """
        input_ids: [B, L]
        attention_mask: [B, L, L]
        """
        input_ids, attention_mask, position_ids = convert_to_kbert(input_ids, attention_mask)
        # position_ids = None
        # print(input_ids)
        # print(attention_mask)
        # print(position_ids)
        # input()
        
        # [B, L, L] => [B, 1, L, L]
        extended_attention_mask = attention_mask.unsqueeze(1)  # .unsqueeze(2)
        # print(input_ids.size())
        # print(extended_attention_mask.size())
        # print(extended_attention_mask.size())
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        
        self.attention_mask = attention_mask[:, :, 0]  # 0 => i-j
        
        return (sequence_output, )
        
    def get_attention_mask(self):
        return self.attention_mask
        

def convert_to_kbert(input_ids, attention_mask):
    """
    input_ids: [B, L]

    ▁[  636
    ]    500
    """
    batch_size, max_seq_length = input_ids.size()
    device = input_ids.device
    
    new_input_ids = []
    new_attention_mask = []
    position_ids = []
    
    for i in range(batch_size):
        seq_length = sum(attention_mask[i])
        
        _input_ids, _mask, _position_ids = convert(input_ids[i, :seq_length], 636, 500, max_seq_length)
        new_input_ids.append(_input_ids)
        new_attention_mask.append(_mask)
        position_ids.append(_position_ids)
    
    new_input_ids = torch.tensor(new_input_ids, dtype=torch.long, device=device)
    new_attention_mask = torch.tensor(new_attention_mask, dtype=torch.long, device=device)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    
    return new_input_ids, new_attention_mask, position_ids
       
       
def convert(ids, start_tag, end_tag, max_seq_length):
    """
    A _[ is B ] C D _[ is E]
    """
    new_ids = []
    types = []
    inside = False

    type_id = 0
    for ch in ids:
        if ch == start_tag:
            inside = True
            type_id = type_id + 1

        elif ch == end_tag:
            inside = False

        else:
            new_ids.append(ch)
            if inside:
                types.append(type_id)
            else:
                types.append(0)

    mask = make_mask(types, max_seq_length)
    position_ids = make_position_ids(types)
        
    new_ids = new_ids + [0] * (max_seq_length - len(new_ids))
    position_ids = position_ids + [0] * (max_seq_length - len(position_ids))
    
    return new_ids, mask, position_ids


def make_mask(types, max_seq_length):
    """
    0, 1, 1, 0, 0, 2, 2, 0
    """
    mask = [[0 for _ in range(max_seq_length)] for _ in range(max_seq_length)]

    max_type = max(types)

    for i in range(max_type+1):
        loc = [l for l, t in enumerate(types) if t == i]
        one(mask, loc, loc)# 每个常识分句各部分彼此都可见

        if i != 0:
            last_word_loc = loc[0] - 1
            while types[last_word_loc] != 0:
                last_word_loc -= 1
            # one(mask, [last_word_loc], loc)
            # one(mask, [last_word_loc], [loc[0]])
            half_one(mask, [last_word_loc], [loc[0]]) # 每个分支句子对主干句子的影响
            # half_one(mask, [last_word_loc], loc)

    return mask


def one(mask, loc1, loc2):
    for i in loc1:
        for j in loc2:
            mask[i][j] = 1
            mask[j][i] = 1
            

def half_one(mask, loc1, loc2):
    """
    [B, (i), L] [B, L, H] => [B, (i), H] => h[:, i]
    
    m[i][j]: hj -> hi 的信息流
    """
    for l1 in loc1:
        for l2 in loc2:
            mask[l1][l2] = 1
            mask[l2][l1] =1
    


def make_position_ids(types):
    """
    0, 1, 1, 0, 0, 2, 2, 0
    """
    pid = 0
    nega_num = 0
    last_state = 0

    positon_ids = []
    for t in types:

        if last_state == 0 and t != 0:
            nega_num = 0
            pid = pid - 1  # for [PAD]

        elif last_state != 0 and t == 0:
            pid = pid - nega_num + 1

        elif last_state != t:  # 同一个词的不同注释
            pid = pid - nega_num
            nega_num = 0

        positon_ids.append(pid)
        pid = pid + 1

        nega_num = nega_num + 1
        last_state = t

    return positon_ids
    
    
    
    
    
    
    
    
        
