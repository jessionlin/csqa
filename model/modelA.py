from .layers import AttentionMerge

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_albert import AlbertPreTrainedModel
from transformers.modeling_albert import AlbertModel


class ModelA(AlbertPreTrainedModel):
    """
    AlBert-AttentionMerge-Classifier

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """
    def __init__(self, config):
        super(ModelA, self).__init__(config)

        self.albert = AlbertModel(config)

        self.att_merge = AttentionMerge(
            config.hidden_size, 1024, 0.1)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def score(self, h1, h2):
        """
        h1, h2: [B, H] => logits: [B, 2]
        """
        logits1 = self.scorer(h1)
        logits2 = self.scorer(h2)
        logits = torch.cat((logits1, logits2), dim=1)
        return logits

    def forward(self, idx, input_ids, attention_mask, token_type_ids, labels):
        """
        input_ids: [B, 2, L]
        labels: [B, ]
        """
        # logits: [B, 2]
        logits = self._forward(idx, input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)

        return loss, right_num, self._to_tensor(idx.size(0), idx.device)

    def _forward(self, idx, input_ids, attention_mask, token_type_ids):
        # [B, 2, L] => [B*2, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )

        # outputs[0]: [B*2, L, H] => [B*2, H]
        h12 = self.att_merge(outputs[0], flat_attention_mask)
        # [B*2, H] => [B*2, 1] => [B, 2]
        logits = self.scorer(h12).view(-1, 2)

        return logits

    def predict(self, idx, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 2]
        """
        return self._forward(idx, input_ids, attention_mask, token_type_ids)

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)
