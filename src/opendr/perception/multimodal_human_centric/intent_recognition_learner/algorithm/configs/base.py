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
from easydict import EasyDict
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.configs.mult_bert import Param
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.data import benchmarks


class ParamManager:

    def __init__(self, args):

        hyper_param, common_param = self._get_config_param()
        benchmark_param = benchmarks[args['benchmark']]

        self.args = EasyDict(
            dict(
                **args,
                **common_param,
                **hyper_param,
                **benchmark_param
            )
        )

    def _get_config_param(self):

        method_args = Param()

        return method_args.hyper_param, method_args.common_param
