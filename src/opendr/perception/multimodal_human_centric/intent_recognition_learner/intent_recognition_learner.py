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

from urllib.request import urlretrieve
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.configs.base import ParamManager
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.data.text_pre import (
    convert_rawtext_to_features,
    tokenizer_factory
)
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.utils.functions import (
    set_output_path,
    set_torch_seed,
    save_results
)
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.backbones.base import ModelManager
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.methods.MULT.manager import MULT
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.target import Category
from torch.utils.data import DataLoader
import torch
import logging
import datetime
import nltk.data
import os
import nltk
nltk.download("punkt", download_dir='./')


_TEXT_BACKBONES = ['bert-base-uncased', 'albert-base-v2', 'prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-small']


class IntentRecognitionLearner(Learner):
    def __init__(
            self,
            text_backbone="bert-base-uncased",
            mode='joint',
            log_path='logs',
            cache_path='cache',
            results_path='results',
            output_path='outputs',
            device='cuda',
            benchmark='MIntRec'):
        super(IntentRecognitionLearner, self).__init__(device=device)
        assert text_backbone in _TEXT_BACKBONES, 'Unsupported text backbone: {}'.format(text_backbone)
        assert mode in ['language', 'joint'], 'Unsupported mode: {}'.format(mode)

        if not torch.cuda.is_available():
            device = 'cpu'

        train_config = {'mode': mode,
                        'text_backbone': text_backbone,
                        'log_path': log_path,
                        'cache_path': cache_path,
                        'results_path': results_path,
                        'output_path': output_path,
                        'device': device,
                        'benchmark': benchmark,
                        'logger_name': 'l'}

        self.train_config = ParamManager(train_config).args
        self.log_path = log_path
        self.logger = self.__set_logger()
        self.train_config.pred_output_path, self.train_config.model_output_path = set_output_path(self.train_config)
        set_torch_seed(self.train_config.seed)
        self.model = ModelManager(self.train_config)
        self.method = MULT(self.train_config, self.model)

        self.tokenizer = None  # placeholder for inference tokenizer
        self.sentence_tokenizer = None

    def fit(self, dataset, val_dataset=None, silent=False, verbose=False):
        """ Performs training on the provided dataset
        :parameter dataset: Object that holds the training set.
        :type dataset: OpenDR Dataset
        :parameter val_dataset: Object that holds the validation set.
        :type val_dataset: OpenDR Dataset
        :parameter silent: Enables training in the silent mode, i.e., only critical output is produced.
        :type silent: bool
        :parameter verbose: Enables verbosity.
        :type verbose: bool
        """
        self.__update_verbosity(silent, verbose)
        self.logger.debug("=" * 30 + " Params " + "=" * 30)
        for k in self.train_config.keys():
            self.logger.debug(f"{k}: {self.train_config[k]}")
        self.logger.debug("=" * 30 + " End Params " + "=" * 30)

        train_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.train_config.train_batch_size,
            num_workers=self.train_config.num_workers,
            pin_memory=True)
        if val_dataset is None:
            val_dataset = dataset
        val_dataloader = DataLoader(val_dataset, batch_size=self.train_config.eval_batch_size,
                                    num_workers=self.train_config.num_workers, pin_memory=True)

        self.logger.info("Starting training...")

        self.method.train(train_dataloader, val_dataloader)

        self.logger.info("Finished training")

    def eval(self, dataset, modality, silent=False, verbose=False, restore_best_model=False):
        """
        Performs evaluation on the test set
        :parameter dataset: dataset used for testing
        :type dataset: OpenDR Dataset
        :parameter modality: Specifies the modality to be used for inference.
        Should either match the current training mode of the learner, or for a learner trained in joint mode,
        any modality can be used for inference.
        :type modality: str
        :parameter silent: If True, run in silent mode, i.e., with only critical output.
        :type silent: bool
        :parameter verbose: If True, provide detailed logs.
        :type verbose: bool
        :parameter restore_best_model: If True, best model obtained on validation set will be loaded from self.output_path.
        If False, current model state will be evaluated.
        :type restore_best_model: bool
        :return: Performance metrics on test set, predicted labels.
        :rtype: dict
        """
        assert modality in ['audio', 'video', 'language', 'joint'], 'Unknown modality: {}'.format(modality)
        assert (
            modality == self.train_config.mode or self.train_config.mode == 'joint'), \
            'Inference on modality {} not supported with mode {}'.format(
            modality, self.train_config.mode)

        self.__update_verbosity(silent, verbose)

        dataloader = DataLoader(dataset, batch_size=self.train_config.test_batch_size,
                                num_workers=self.train_config.num_workers, pin_memory=True)

        self.logger.info("Started testing...")
        outputs = self.method.test(dataloader, modality, restore_best_model)

        save_results(self.train_config, outputs, suff=modality, results_file_name=modality + '_results.csv')

        self.logger.info("Finished testing...")

        return outputs

    def infer(self, batch, modality='language'):
        """
        Splits the input text into sentences, process each sentence independently.
        If a sentence is > max_sequence_length, process in sliding window manner.
        :parameter batch: Input data
        :type batch: dict
        :parameter modality: Specifiec inference modality
        :type modality: str
        :return: Predicted class label with confidence score for each sentence in the input
        :rtype: list of opendr.data.Category
        """
        assert modality in ['language'], 'Unsupported modality {}'.format(modality)
        if self.sentence_tokenizer is None:
            self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        out = []
        for sentence in self.sentence_tokenizer.tokenize(batch['text']):
            text_feats = self.__process_raw_text(sentence)
            pred = self.method.infer(text_feats, modality=modality)
            out.append(Category(int(pred[0]), confidence=pred[1]))
        return out

    def save(self, path):
        """ Saves current state of the model to the given path

        :parameter path: Path where to save the model
        :type path: str
        """
        self.logger.info("Saving model...")
        state_dict = {
            'state_dict': self.model.model.state_dict(),
            'config': self.train_config
        }
        torch.save(state_dict, path)
        self.logger.info("Save model.")

    def load(self, path):
        """ Loads model chekpoint from specified path

        :parameter path: Path to the checkpoint
        :type path: str
        """
        self.logger.info("Loading model from {}...".format(path))
        checkpoint = torch.load(path, map_location=self.train_config.device)
        self.model.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.logger.info("Successfully loaded model.")

    def download(self, path):
        server_url = os.path.join(OPENDR_SERVER_URL,
                                  'perception',
                                  'multimodal_human_centric',
                                  'intent_recognition_learner')

        if 'prajjwal1/' in self.train_config.text_backbone:
            model_name = self.train_config.text_backbone.split('/')[1]
        else:
            model_name = self.train_config.text_backbone
        weights_url = os.path.join(server_url, '{}.pth'.format(model_name))
        urlretrieve(weights_url, path)

    def trim(self, modality='language'):
        """ Converts multimodal model to unimodal for inference
        :parameter modality: Specifies which modality to trim the model to: language (recommended) | audio | video
        :type modality: str
        """
        self.logger.info("Converting model to unimodal...")
        if modality == self.train_config.mode:
            return

        curr_weights = self.model.model.state_dict()
        self.train_config.mode = modality
        self.model = ModelManager(self.train_config)
        self.method = MULT(self.train_config, self.model)
        self.model.model.load_state_dict(curr_weights, strict=False)
        self.logger.info("Finished converting model to unimodal...")

    def optimize(self):
        return NotImplementedError

    def reset(self):
        return NotImplementedError

    def __process_raw_text(self, raw_text):
        """
        Tokenizes raw text.
        """
        if self.tokenizer is None:
            self.tokenizer = tokenizer_factory(self.train_config.text_backbone, self.train_config.cache_path)
        features = convert_rawtext_to_features(raw_text, self.train_config.max_seq_length_text, self.tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        text_feats = torch.tensor(features_list)
        text_feats = text_feats.to(self.train_config.device)
        return text_feats

    def __update_verbosity(self, silent, verbose):
        """
        Updates verbosity level of the logger.
        """
        if silent:
            level = logging.CRITICAL
        elif verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def __set_logger(self):
        """
        Sets up global logger for training.
        """
        if not os.path.exists(self.train_config.log_path):
            os.makedirs(self.train_config.log_path)
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.train_config.logger_name = self.train_config.logger_name + '_' + \
            f"{self.train_config.mode}_{self.train_config.alpha}_{self.train_config.seed}_{time}"
        logger = logging.getLogger(self.train_config.logger_name)
        logger.setLevel(logging.INFO)

        log_path = os.path.join(self.train_config.log_path, self.train_config.logger_name + '.log')
        fh = logging.FileHandler(log_path)
        fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        return logger
