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

# general imports
import torch
from torch.utils.data import DataLoader
import os
import json
from torch.utils.tensorboard import SummaryWriter
from urllib.request import urlretrieve
import librosa

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Video, Timeseries
from opendr.engine.datasets import DatasetIterator
from opendr.engine.target import Category
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR imports
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm import (data,
                                                                                              models,
                                                                                              trainer,
                                                                                              spatial_transforms as transforms,
                                                                                              data_utils
                                                                                              )

# constants
PRETRAINED_MODEL = ['ia_zerodrop']

__all__ = []


class AudiovisualEmotionLearner(Learner):
    def __init__(self,
                 num_class=8,
                 seq_length=15,
                 fusion='ia',
                 mod_drop='zerodrop',
                 pretr_ef=None,
                 lr=0.04,
                 lr_steps=[40, 55, 65, 70, 200, 250],
                 momentum=0.9,
                 dampening=0.9,
                 weight_decay=1e-3,
                 iters=100,
                 batch_size=8,
                 n_workers=4,
                 device='cpu',
                 ):
        super(AudiovisualEmotionLearner,
              self).__init__(batch_size=batch_size,
                             device=device)
        assert fusion in ['ia', 'it', 'lt'], 'Unknown modality fusion type'
        assert mod_drop in ['nodrop', 'noisedrop', 'zerodrop'], 'Unknown modlaity dropout type'

        self.model = models.MultiModalCNN(num_classes=num_class, fusion=fusion, seq_length=seq_length, pretr_ef=pretr_ef)

        self.num_class = num_class
        self.fusion = fusion

        self.lr = lr
        self.lr_steps = lr_steps

        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        self.n_iters = iters
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.mod_drop = mod_drop

        self.seq_length = seq_length

    def _validate_x1(self, x):
        if not isinstance(x, Timeseries):
            msg = 'The 1st element returned by __getitem__ must be an instance of `engine.data.Timeseries` class\n' +\
                  'Received an instance of type: {}'.format(type(x))
            raise TypeError(msg)

    def _validate_x2(self, x):
        if not isinstance(x, Video):
            msg = 'The 2nd element returned by __getitem__ must be an instance of `engine.data.Video` class\n' +\
                  'Received an instance of type: {}'.format(type(x))
            raise TypeError(msg)

        if x.data.shape[0] != 3:
            msg = 'The first dimension of data produced by dataset must be 3\n' +\
                  'Received input of shape: {}'.format(x.data.shape)
            raise ValueError(msg)

        if x.data.shape[1] != self.seq_length:
            msg = 'The temporal dimension of data does not match specified sequence length of the model\n' +\
                  'Received input with dimension: {} and sequence length is: {}.'.format(x.data.shape[1], self.seq_length)
            raise ValueError(msg)

    def _validate_y(self, y):
        if not isinstance(y, Category):
            msg = 'The 2nd element returned by __getitem__ must be an instance of `engine.target.Cateogry` class\n' +\
                  'Received an instance of type: {}'.format(type(y))
            raise TypeError(msg)

    def _validate_dataset(self, dataset):
        """
        This internal function is used to perform basic validation of the data dimensions
        """
        if dataset is None:
            return

        if not isinstance(dataset, data.RavdessDataset):
            if not isinstance(dataset, DatasetIterator):
                msg = 'Dataset must be an instance of engine.datasets.DatasetIterator class\n' +\
                      'Received an instance of type: {}'.format(type(dataset))
                raise TypeError(msg)
            else:
                x1, x2, y = dataset.__getitem__(0)
                self._validate_x1(x1)
                self._validate_x2(x2)
                self._validate_y(y)

    def fit(self, dataset, val_dataset=None, logging_path='logs/', silent=False, verbose=True,
            eval_mode='audiovisual', restore_best=False):
        """
        Method to train the audiovisual emotion recognition model

        :param dataset: training dataset object
        :type dataset: engine.datasets.DatasetIterator
        :param val_dataset: validation samples, default to None

                        if available, `val_set` is used to select
                        the best checkpoint for final model
        :type val_dataset: engine.datasets.DatasetIterator

        :param logging_path: path to save checkpoints
                             and log data, default to "logs/"
        :type logging_path: string
        :param silent: disable performance printing, default to False
        :type silent: bool
        :param verbose: enable the performance printing, default to True
        :type verbose: bool

        :return: the best accuracy on validation set
        :rtype: float
        """
        self._validate_dataset(dataset)
        self._validate_dataset(val_dataset)
        assert eval_mode in ['audiovisual', 'noisyaudio', 'noisyvideo', 'onlyaudio', 'onlyvideo']

        if isinstance(dataset, data.RavdessDataset):
            train_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      pin_memory=self.device == 'cuda',
                                      num_workers=self.n_workers,
                                      shuffle=True)
        else:
            train_loader = DataLoader(data.DataWrapper(dataset),
                                      batch_size=self.batch_size,
                                      pin_memory=self.device == 'cuda',
                                      num_workers=self.n_workers,
                                      shuffle=True)

        if val_dataset is None:
            val_loader = None
        elif isinstance(val_dataset, data.RavdessDataset):
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.n_workers,
                                    pin_memory=self.device == 'cuda',
                                    shuffle=False)
        else:
            val_loader = DataLoader(data.DataWrapper(val_dataset),
                                    batch_size=self.batch_size,
                                    num_workers=self.n_workers,
                                    pin_memory=self.device == 'cuda',
                                    shuffle=False)

        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        tensorboard_logger = SummaryWriter(logging_path)
        self.model = self.model.to(self.device)
        metrics = trainer.train(self.model, train_loader, val_loader, self.lr, self.momentum, self.dampening,
                                self.weight_decay, self.n_iters, logging_path, self.lr_steps,
                                self.mod_drop, self.device, silent, verbose,
                                tensorboard_logger, eval_mode, restore_best)

        if tensorboard_logger is not None:
            tensorboard_logger.close()
        return metrics

    def eval(self, dataset, silent=False, verbose=True, mode='audiovisual'):
        """
        This method is used to evaluate the performance of a given set of data

        :param dataset: object that holds the set of samples to evaluate
        :type dataset: engine.datasets.DatasetIterator
        :param mode: testing mode
        :type mode: string
        :return: a dictionary that contains `cross_entropy` and `acc` as keys
        :rtype: dict
        """
        self._validate_dataset(dataset)
        if isinstance(dataset, data.RavdessDataset):
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.n_workers,
                                pin_memory=self.device == 'cuda',
                                shuffle=False)
        else:
            loader = DataLoader(data.DataWrapper(dataset),
                                batch_size=self.batch_size,
                                num_workers=self.n_workers,
                                pin_memory=self.device == 'cuda',
                                shuffle=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        L = torch.nn.CrossEntropyLoss()
        loss, acc = trainer.val_one_epoch(-1, loader, self.model, L, mode=mode, device=self.device,
                                          silent=silent, verbose=verbose)
        if not silent and verbose:
            print('Loss: {}, Accuracy: {}'.format(loss, acc))

        return {'cross_entropy': loss, 'acc': acc}

    def _process_video(self, video_path, target_time=3.6, input_fps=30, save_frames=15, target_im_size=224):
        """
        This function preprocesses input video file for inference
        Parameters
        ----------
        video_path : str
            path to video file.
        target_time : float, optional
            Target time of processed video file in seconds. The default is 3.6.
        input_fps : int, optional
            Frames Per Second of input video file. The default is 30.
        save_frames : int, optional
            Length of target frame sequence. The default is 15.
        target_im_size : int, optional
            Target width and height of each frame. The default is 224.

        Returns
        -------
        numpy_video: numpy.array
                     N frames as numpy array

        """
        numpy_video = data_utils.preprocess_video(video_path, target_time, input_fps, save_frames,
                                                  target_im_size, device=self.device)
        video_scale = 255
        video_transform = transforms.Compose([
                              transforms.ToTensor(video_scale)])
        video = [video_transform(img) for img in numpy_video]
        video = torch.stack(video, 0).permute(1, 0, 2, 3)

        return video

    def _process_audio(self, audio_path, target_time=3.6, sr=22050):
        """
        This function preprocesses an audio file for inference

        Parameters
        ----------
        audio_path : str
            Path to audio file.
        target_time : int, optional
            Target duration of audio. The default is 3.6.
        sr : int, optional
            Sampling rate of audio. The default is 22050.

        Returns
        -------
        y : numpy array
            audio file saved as numpy array.
        """
        y = data_utils.preprocess_audio(audio_path, sr, target_time)
        mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=10)
        return mfcc

    def load_inference_data(self, audio_path, video_path, target_time=3.6, sr=22050, input_fps=30, save_frames=15,
                            target_im_size=224):
        video = Video(self._process_video(video_path, target_time, input_fps, save_frames, target_im_size))
        audio = Timeseries(self._process_audio(audio_path, target_time, sr))
        return audio, video

    def infer(self, audio, video):
        """
        This method is used to generate prediction given Audio and Visual data

        :param video: video of a fronal view of a face
        :type video: engine.data.Video
        :param audio: audio features to generate class prediction
        :type audio: engine.data.Timeseries
        :return: predicted label
        :rtype: engine.target.Category

        """
        self._validate_x1(audio)
        self._validate_x2(video)
        self.model.to(self.device)
        self.model.eval()

        video = torch.tensor([video.data]).permute(0, 2, 1, 3, 4)
        video = video.reshape(video.shape[0]*video.shape[1], video.shape[2],
                              video.shape[3], video.shape[4]).to(self.device)

        audio = torch.tensor(audio.data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob_prediction = torch.nn.functional.softmax(self.model(audio, video).flatten(), dim=0)
            class_prediction = prob_prediction.argmax(dim=-1).cpu().item()

        prediction = Category(class_prediction, confidence=prob_prediction[class_prediction].cpu().item())

        return prediction

    def pred_to_label(self, prediction):
        """
        This function converts the numeric class value to huamn-interpretable emotion label for RAVDESS dataset
        """
        assert self.num_class == 8, 'Unknown emotion class vocabulary for given number of classes'

        NUM_2_CLASS = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
        return NUM_2_CLASS[prediction.data]

    def save(self, path, verbose=True):
        """
        This function is used to save the current model given a directory path.
        Metadata and model weights are saved under `path/metadata.json`
        and path/model_weights.pt`
        :param path: path to the directory that the model will be saved
        :type path: str
        :param verbose: default to True
        :type verbose: bool

        """
        if not os.path.exists(path):
            os.makedirs(path)

        model_weight_file = os.path.join(path, 'model_weights.pt')
        metadata_file = os.path.join(path, 'metadata.json')

        metadata = {'framework': 'pytorch',
                    'model_paths': ['model_weights.pt'],
                    'format': 'pt',
                    'has_data': False,
                    'inference_params': {},
                    'optimized': False,
                    'optimimizer_info': {}
                    }

        try:
            torch.save(self.model.cpu().state_dict(), model_weight_file)
            if verbose:
                print('Model weights saved to {}'.format(model_weight_file))
        except Exception as error:
            raise error

        try:
            fid = open(metadata_file, 'w')
            json.dump(metadata, fid)
            fid.close()

            if verbose:
                print('Model metadata saved to {}'.format(metadata_file))

        except Exception as error:
            raise error

        return True

    def load(self, path, verbose=True):
        """
        This function is used to load a pretrained model that
        has been saved with .save(), given the path to the directory.
        `path/metadata.json` and `path/model_weights.pt` should exist
        :param path: path to the saved location
        :type path: str
        :param verbose: defaul to True
        :type verbose: bool

        """

        if not os.path.exists(path):
            raise FileNotFoundError('Directory "{}" does not exist'.format(path))

        if not os.path.isdir(path):
            raise ValueError('Given path "{}" is not a directory'.format(path))

        metadata_file = os.path.join(path, 'metadata.json')
        assert os.path.exists(metadata_file),\
            'Metadata file ("metadata.json")' +\
            'does not exist under the given path "{}"'.format(path)

        fid = open(metadata_file, 'r')
        metadata = json.load(fid)
        fid.close()

        model_weight_file = os.path.join(path, metadata['model_paths'][0])
        assert os.path.exists(model_weight_file),\
            'Model weights "{}" does not exist'.format(model_weight_file)

        self.model.cpu()
        self.model.load_state_dict(torch.load(model_weight_file,
                                   map_location=torch.device('cpu')))

        if verbose:
            print('Pretrained model is loaded successfully')

    def download(self, path):
        """
        This function is used to download a pretrained model for the audiovisual emotion recognition task
        Calling load(path) after this function will load the downloaded model weights

        :param path: path to the saved location. Under this path `model_weights.pt` and `metadata.json`
                     will be downloaded so different paths for different models should be given to avoid
                     overwriting previously downloaded model
        :type path: str
        """
        print('Downloading pre-trained model trained on RAVDESS dataset under  CC BY-NC-SA 4.0 license')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if self.fusion + '_' + self.mod_drop in PRETRAINED_MODEL:
            assert self.num_class == 8,\
                'For pretrained audiovisual emotionrecognition model, `num_cluss` must be 8'

            server_url = os.path.join(OPENDR_SERVER_URL,
                                      'perception',
                                      'multimodal_human_centric',
                                      'audiovisual_emotion_learner')

            model_name = '{}_{}_{}'.format('av_emotion', self.fusion, self.mod_drop)

            metadata_url = os.path.join(server_url, '{}.json'.format(model_name))
            metadata_file = os.path.join(path, 'metadata.json')
            urlretrieve(metadata_url, metadata_file)

            weights_url = os.path.join(server_url, '{}.pt'.format(model_name))
            weights_file = os.path.join(path, 'model_weights.pt')
            urlretrieve(weights_url, weights_file)
            print('Pretrained model downloaded to the following directory\n{}'.format(path))
        else:
            raise UserWarning('No pretrained model for fusion "{}" and modality drop "{}"'.format(self.fusion, self.mod_drop))

    def optimize(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
