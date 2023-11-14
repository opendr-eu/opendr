# active_face_recognition module

The *active_face_recognition* module contains the *ActiveFaceRecognitionLearner* class, which inherits from the abstract 
class *LearnerRL*.

### Class ActiveFaceRecognitionLearner
Bases: `engine.learners.LearnerRL`

The *ActiveFaceRecognitionLearner* is an agent that can be used to train a quadrotor robot equipped with an RGB camera 
to maximize the confidence of OpenDR's FaceRecognition module.

The [ActiveFaceRecognitionLearner](../../src/opendr/perception/active_perception/active_face_recognition/active_face_recognition_learner.py) class has the 
following public methods:

#### `ActiveFaceRecognitionLearner` constructor

Constructor parameters:
- **lr**: *float, default=3e-4*\
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=5e6*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*\
  Specifies the batch size during training.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies per how many training steps a checkpoint should be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint to load.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm stores log files and saves checkpoints.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **n_steps**: *int, default=6400*\
  Specifies the number of steps to run environment per update.
- **gamma**: *float, default=0.9*\
  Discount factor for future rewards.
- **clip_range**: *float, default=0.1*\
  Clip policy updates.
- **target_kl**: *float, default=0.1*\
  KL Divergence update threshold.

#### `ActiveFaceRecognitionLearner.fit`
```python
ActiveFaceRecognitionLearner.fit(self, logging_path, verbose)
```

Train the agent on the environment.

Parameters:

- **logging_path**: *str, default='./'*\
  Path for logging and checkpointing.
- **verbose**: *bool, default=True*\
  Enable verbosity.


#### `ActiveFaceRecognitionLearner.eval`
```python
ActiveFaceRecognitionLearner.eval(self, num_episodes, deterministic)
```
Evaluate the agent on the specified environment.

Parameters:

- **num_episodes**: *int, default=10*\
  Number of evaluation episodes to run.
- **deterministic**: *bool, default=False*\
  Use deterministic actions from the policy.


#### `ActiveFaceRecognitionLearner.infer`
```python
ActiveFaceRecognitionLearner.infer(self, observation, deterministic)
```
Performs inference on a single observation.

Parameters:

- **observation**: *engine.Image, default=None*\
  Single observation.
- **deterministic**: *bool, default=False*\
  Use deterministic actions from the policy

#### `ActiveFaceRecognitionLearner.save`
```python
ActiveFaceRecognitionLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `ActiveFaceRecognitionLearner.load`
```python
ActiveFaceRecognitionLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.



#### `ActiveFaceRecognitionLearner.download`
```python
ActiveFaceRecognitionLearner.download(self, path)
```
Downloads a pretrained model to the path provided.

Parameters:

- **path**: *str*\
  Path to download model.

### Running the environment

The environment is provided with a [world](../../src/opendr/perception/active_perception/active_face_recognition/simulation/worlds/active_face_recognition.wbt)
that needs to be opened with Webots version 2023b in order to demonstrate the active face recognition learner.

Once the world is opened and the simulation is running, you can run a script utilizing ActiveFaceRecognitionLearner \
by setting WEBOTS_HOME environment variable:

`export WEBOTS_HOME=/usr/local/webots`

and then run the desired script, e.g. demo.py by:

`$WEBOTS_HOME/webots-controller /path/to/script/demo.py`


### Examples

Training in Webots environment:

```python
from opendr.perception.active_perception.active_face_recognition import ActiveFaceRecognitionLearner

learner = ActiveFaceRecognitionLearner(n_steps=1024)
learner.fit(logging_path='./active_face_recognition_tmp')
```


Evaluating a pretrained model:

```python
from opendr.perception.active_perception.active_face_recognition import ActiveFaceRecognitionLearner

learner = ActiveFaceRecognitionLearner()
path = './'
learner.download(path)
learner.load("./active_fr.zip")
rewards = learner.eval(num_episodes=10, deterministic=False)

print(rewards)
```


Running inference on a pretrained model:

```python
from opendr.perception.active_perception.active_face_recognition import ActiveFaceRecognitionLearner

learner = ActiveFaceRecognitionLearner()
path = './'
learner.download(path)
learner.load("./active_fr.zip")

obs = learner.env.reset()
while True:
    action, _ = learner.infer(obs)
    obs, reward, done, info = learner.env.step(action)
    if done:
        obs = learner.env.reset()
```


### Performance Evaluation

TABLE 1: Speed (FPS) for inference on various platforms.

|                 | TX2    | XavierNX | RTX 2070 Super |
| --------------- |--------|----------|----------------|
| FPS Evaluation  | 425.27 | 512.48   | 683.25         |

TABLE 2: Platform compatibility evaluation.

| Platform                                     | Test results |
|----------------------------------------------| ------------ |
| x86 - Ubuntu 20.04 (bare installation - CPU) | Pass         |
| x86 - Ubuntu 20.04 (bare installation - GPU) | Pass         |
| x86 - Ubuntu 20.04 (pip installation)        | Pass         |
| x86 - Ubuntu 20.04 (CPU docker)              | Pass         |
| x86 - Ubuntu 20.04 (GPU docker)              | Pass         |
| NVIDIA Jetson TX2                            | Pass         |
| NVIDIA Jetson Xavier NX                      | Pass         |

