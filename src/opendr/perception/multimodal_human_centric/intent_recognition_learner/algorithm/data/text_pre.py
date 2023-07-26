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


import csv
import os
import sys
from transformers import BertTokenizer, AlbertTokenizer

__all__ = ['TextDataset']


def tokenizer_factory(text_backbone, cache_dir='cache'):
    if text_backbone.startswith('bert') or text_backbone.startswith('prajjwal1/bert'):
        tokenizer = BertTokenizer.from_pretrained(text_backbone, do_lower_case=True, cache_dir=cache_dir)
    elif text_backbone.startswith('albert'):
        tokenizer = AlbertTokenizer.from_pretrained(text_backbone, do_lower_case=True, cache_dir=cache_dir)
    return tokenizer


class TextDataset:

    def __init__(self, text_backbone, base_attrs, split):

        self.base_attrs = base_attrs

        if text_backbone.startswith('bert') or text_backbone.startswith('albert') or text_backbone.startswith('prajjwal1/bert'):
            self.feats = self._get_feats(text_backbone, base_attrs, split)
        else:
            raise Exception('Error: inputs are not supported text backbones.')

    def _get_feats(self, text_backbone, base_attrs, split):

        processor = DatasetProcessor()
        examples = processor.get_examples(base_attrs['data_path'], split)
        feats = self._get_bert_feats(text_backbone, examples, base_attrs)
        return feats

    def _get_bert_feats(self, text_backbone, examples, base_attrs):

        max_seq_length = base_attrs["benchmarks"]['max_seq_length_text']
        tokenizer = tokenizer_factory(text_backbone, self.base_attrs['cache_path'])
        features = convert_examples_to_features(examples, max_seq_length, tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]

        return features_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, split):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(split))), split)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == 'dev':
            set_type = 'train'
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[3]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples


def convert_rawtext_to_features(text, max_seq_length, tokenizer):
    features = []
    tokens_a = tokenizer.tokenize(text)
    examples = []
    window = min(max(0, max_seq_length - 2 - 1), 5)
    while len(tokens_a) > max_seq_length - 2:
        examples.append(tokens_a[:max_seq_length - 2])
        tokens_a = tokens_a[max_seq_length - 2 - window:]

    examples.append(tokens_a)
    for tokens_a in examples:

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
        )
    return features


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # tokens = tokens_a

        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
