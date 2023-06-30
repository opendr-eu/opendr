import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AlbertModel

__all__ = ['BERTEncoder', 'Albertv2Encoder']


def model_factory(args):
    if args.text_backbone.startswith('albert'):
        return AlbertModel.from_pretrained(args.text_backbone, cache_dir=args.cache_path)
    elif args.text_backbone.startswith('bert') or args.text_backbone.startswith('prajjwal1/bert'):
        return BertModel.from_pretrained(args.text_backbone, cache_dir=args.cache_path)


class LanguageModel(nn.Module):
    def __init__(self, args):
        super(LanguageModel, self).__init__()
        self.model = model_factory(args)
        print(args.text_backbone)

    def forward(self, text_feats):
        input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
