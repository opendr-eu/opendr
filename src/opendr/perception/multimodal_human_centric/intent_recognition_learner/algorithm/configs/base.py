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
