# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from opendr.perception.multimodal_human_centric import IntentRecognitionLearner
import argparse

LABELS = [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort',
            'Care', 'Inform', 'Advise', 'Arrange', 'Introduce',
            'Leave', 'Prevent', 'Greet', 'Ask for help'
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_dir", help="Path to the directory where pretrained models are saved", type=str, default="./pretrained_models")
    parser.add_argument("--text_backbone",
                        help="Text backbone: ['bert-base-uncased' | 'albert-base-v2' | \
                            'bert-small' | 'bert-mini' | 'bert-tiny']",
                        type=str, default="bert-base-uncased")
    parser.add_argument("--cache_path", help="Cache path for tokenizer files", type=str, default="cache")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.text_backbone == 'bert-small':
        text_backbone = 'prajjwal1/bert-small'
    elif args.text_backbone == 'bert-mini':
        text_backbone = 'prajjwal1/bert-mini'
    elif args.text_backbone == 'bert-tiny':
        text_backbone = 'prajjwal1/bert-tiny'
    else:
        text_backbone = args.text_backbone

    modality = 'language'

    learner = IntentRecognitionLearner(text_backbone=text_backbone, mode=modality,
                                       device=args.device, log_path='logs',
                                       cache_path=args.cache_path, results_path='result',
                                       output_path='outputs')
    if not os.path.exists('{}/{}.pth'.format(args.model_dir, args.text_backbone)):
        learner.download('{}/{}.pth'.format(args.model_dir, args.text_backbone))
    learner.load('{}/{}.pth'.format(args.model_dir, args.text_backbone))

    while True:
        raw_text = input('Enter text: ')
        pred = learner.infer({'text': raw_text}, modality='language')
        for utt in pred:
            print(LABELS[utt.data], utt.confidence)
