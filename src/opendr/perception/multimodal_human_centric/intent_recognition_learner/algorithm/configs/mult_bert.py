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


class Param():

    def __init__(self):

        self.common_param = self._get_common_parameters()
        self.hyper_param = self._get_hyper_parameters()

    def _get_common_parameters(self):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end').
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation.
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'need_aligned': False,
            'eval_monitor': 'f1',
            'train_batch_size': 32,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            dst_feature_dims (int): The destination dimensions (assume d(l) = d(v) = d(t)).
            nheads (int): The number of heads for the transformer network.
            n_levels (int): The number of layers in the network.
            attn_dropout (float): The attention dropout.
            attn_dropout_v (float): The attention dropout for the video modality.
            attn_dropout_a (float): The attention dropout for the audio modality.
            relu_dropout (float): The relu dropout.
            embed_dropout (float): The embedding dropout.
            res_dropout (float): The residual block dropout.
            output_dropout (float): The output layer dropout.
            text_dropout (float): The dropout for text features.
            grad_clip (float): The gradient clip value.
            attn_mask (bool): Whether to use attention mask for Transformer.
            conv1d_kernel_size_l (int): The kernel size for temporal convolutional layers (text modality).
            conv1d_kernel_size_v (int):  The kernel size for temporal convolutional layers (video modality).
            conv1d_kernel_size_a (int):  The kernel size for temporal convolutional layers (audio modality).
            lr (float): The learning rate of backbone.
            seed (int): random seed.
            alpha (float): Knowledge transfer loss coefficient.
            betha (float): Unimodal branches task-specific loss coefficient.
            gamma (float): Multimodal branch task-specific loss coefficient.
            t (int): Temperature used for distillation.
            'attentiondist' (bool): Perform attention distillation or not.
            'attentiondist' (bool): Perform attention together with decision-level distillation or not.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'dst_feature_dims': 120,
            'nheads': 8,
            'n_levels': 8,
            'attn_dropout': 0.0,
            'attn_dropout_v': 0.2,
            'attn_dropout_a': 0.2,
            'relu_dropout': 0.0,
            'embed_dropout': 0.1,
            'res_dropout': 0.0,
            'output_dropout': 0.2,
            'text_dropout': 0.4,
            'grad_clip': 0.5,
            'attn_mask': True,
            'conv1d_kernel_size_l': 5,
            'conv1d_kernel_size_v': 1,
            'conv1d_kernel_size_a': 1,
            'lr': 0.00003,
            'seed': 0,
            'alpha': 0.5,
            'betha': 1,
            'gamma': 1,
            't': 2,
            'attentiondist': False,
            'attentiondistboth': False,
            'num_workers': 0,
            'gpu_id': 0
        }
        return hyper_parameters
