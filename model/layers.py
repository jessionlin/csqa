import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMerge(nn.Module):
    """
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.

        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context


class TextCNN(nn.Module):
    def __init__(self, input_size):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 768, (width, input_size))
            for width in [2, ]
        ])
        # self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, values, a):
        values = values.unsqueeze(1)  # (b, 1, l, h)
        values = [F.relu(conv(values)).squeeze(3) for conv in self.convs]
        values = [F.max_pool1d(value, value.size(2)).squeeze(2) for value in values]
        values = torch.cat(values, 1)
        return values


class WW_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(WW_Attention, self).__init__()
        # self.W1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.bias = nn.Parameter(torch.Tensor(1))
        # self.bias.data.zero_()
        self.score_ = nn.Sequential(
            nn.Linear(2*768, 1),
            nn.ReLU()
        )

    def score(self, h1, h2):
        h = torch.cat((h1, h2), dim=1)
        # (b, 2h, ) -> (b, h, ) -> (b, 1, h)
        return self.score_(h).unsqueeze(1)

    def forward(self, words, diff):
        """
        words: (b, l, h)
        diff:  (b, l)
        """
        scores = []  # [(b, 1, h)]
        for i in range(words.size(1)):
            for j in range(i+1, min(words.size(1), i+6)):
                s = self.score(words[:, i], words[:, j])  # (b, 1, h)
                # s = s * diff[:, i].view(-1, 1, 1).type(torch.float)  # (b, 1, 1)
                scores.append(s)
        scores = torch.cat(scores, dim=1)  # (b, ll, h)
        scores = torch.max(scores, dim=1)[0]  # (b, h)
        return scores


def attention(keys, values, query, dropout_prob):
    """
    keys:   (b, l, h)
    values: (b, l, h)
    query:  (b, h)
    """
    attention_probs = keys.bmm(query.unsqueeze(2)) / math.sqrt(query.size(1))

    attention_probs = F.softmax(attention_probs, dim=1)
    attention_probs = F.dropout(attention_probs, p=dropout_prob)

    context = torch.sum(attention_probs*values, dim=1)
    return context


# class AttentionMerge(nn.Module):
#     def __init__(self, input_size, attention_size, dropout_prob):
#         super(AttentionMerge, self).__init__()
#         self.dropout_prob = dropout_prob

#     def forward(self, values):
#         """
#         (b, l, h) -> (b, h)
#         """
#         query = torch.mean(values, dim=1)
#         return attention(values, values, query, self.dropout_prob)


def get_weight(v1, v2, W, b):
    """
    v1: (b, l1, h)
    v2: (b, l2, h)
    W: (h, h)
    b: (1, )
    return (b, l1, l2)
    """

    # print("hidden is {}".format(v1.size()))
    # print("result1 is {}".format((v1 @ W).tolist()[0][0][:5]))
    # print("result2 is {}".format(v2.permute(0, 2, 1).tolist()[0][0][:5]))
    # print("result3 is {}".format((v1 @ W).matmul(v2.permute(0, 2, 1)).tolist()[0][0][:5]))
    # print("size is {}".format(torch.matmul((v1 @ W),v2.permute(0, 2, 1)).size()))
    # return torch.matmul((v1 @ W),v2.permute(0, 2, 1)) + b
    return (v1 @ W).bmm(v2.permute(0, 2, 1)) + b
    # return (v1 @ W + b).bmm(v2.permute(0, 2, 1))


class DualAttention(nn.Module):
    """
    weight = h1 W h2
    通过weight重构h1和h2，注意h1和h2的形状将会交换
    """
    def __init__(self, hidden_size):
        super(DualAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(0, 0.001)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden1, hidden2):
        """
        hidden1: (b, s1, h)
        hidden2: (b, s2, h)
        """

        weight = get_weight(hidden1, hidden2, self.W, self.bias)  # (b, s1, s2)
        # print("weight is {}".format(weight))
        # (b, s1, s2) -> (b, s2, s1) X (b, s1, h) -> (b, s2, h)
        hidden_1 = F.softmax(weight, dim=1).permute(0, 2, 1).bmm(hidden1)
        hidden_2 = F.softmax(weight, dim=2).bmm(hidden2)
        return hidden_1, hidden_2


# class SuAttentionMerge(nn.Module):
#     def __init__(self, hidden_size, dropout_prob):
#         super(SuAttentionMerge, self).__init__()
#         self.att_merge = AttentionMerge(hidden_size, 256, dropout_prob)
#         self.hidden_layer = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, H1, H2):
#         h1 = self.att_merge(H1)
#         h2 = self.att_merge(H2)
#         context1 = self.attention(H1, h2)
#         context2 = self.attention(H2, h1)
#         return context1, context2

#     def attention(self, values, query):
#         keys = self.hidden_layer(values)
#         keys = torch.tanh(keys)
#         attention_probs = keys @ query / math.sqrt(query.size(1))

#         attention_probs = F.softmax(attention_probs, dim=1)
#         attention_probs = self.dropout(attention_probs)

#         context = torch.sum(attention_probs*values, dim=1)
#         return context


class OptionAttention(nn.Module):
    """
    Zhu H, Wei F, Qin B, et al. Hierarchical attention flow for multiple-choice
    reading comprehension[C]//Thirty-Second AAAI Conference on Artificial
    Intelligence. 2018.
    """
    def __init__(self, hidden_size):
        super(OptionAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2, option3):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight3 = get_weight(option1, option3, self.W, self.bias).unsqueeze(-1)
        weight = torch.cat((weight2, weight3), dim=-1)  # (b, s1, s2, 2)
        weight = F.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2 + weight[:, :, :, 1] @ option3
        return option1_

    def forward(self, option1, option2, option3):
        option1_ = self.retain(option1, option2, option3)
        option2_ = self.retain(option2, option1, option3)
        option3_ = self.retain(option3, option1, option2)

        option1_ = option1 - option1_
        option2_ = option2 - option2_
        option3_ = option3 - option3_
        return option1_, option2_, option3_


class DoubletoScore(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(DoubletoScore, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size1, hidden_size2))
        self.W.data.normal_(mean=0.0, std=0.02)

    def forward(self, h1, h2):
        """
        h1: (b, h1)
        h2: (b, h2)
        h1 W h2 -> (b, 1)
        """
        score = h1.unsqueeze(1) @ self.W   # (b, 1, h1) @ (h1, h2) -> (b, 1, h2)
        score = score.bmm(h2.unsqueeze(2)) # (b, 1, h2) X (b, h2, 1) -> (b, 1, 1)
        return score.view(-1, 1)           # (b, 1)


class GateCat(nn.Module):
    def __init__(self, hidden_size):
        super(GateCat, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        gate = self.gate(torch.cat((h1, h2), dim=1))
        return h1 * gate + h1 * (1-gate)
