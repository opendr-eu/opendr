'''
This code is based on https://github.com/thuiar/MIntRec under MIT license:

MIT License

Copyright (c) 2022 Hanlei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''


from torch import nn
from transformers import BertModel, AlbertModel


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
