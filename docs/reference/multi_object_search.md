# multi_object_search module

The *multi_object_search* module contains the *MultiObjectSearchRLLearner* class, which inherits from the abstract class *LearnerRL*.

### Class MultiObjectSearchRLLearner
Bases: `engine.learners.LearnerRL`

The *MultiObjectSearchRLLearner* class is an RL agent that can be used to train wheeled robots for combining short-horizon control with long horizon reasoning into a single policy.
Originally published in [[1]](#multi_object_search), Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces,(https://arxiv.org/abs/2205.11384).

The [MultiObjectSearchRLLearner](/src/opendr/control/multi_object_search/multi_object_search_learner.py) class has the following public methods:

#### `MultiObjectSearchRLLearner` constructor
MultiObjectSearchRLLearner(self, env, lr, ent_coef, clip_range, gamma, n_steps, n_epochs, iters, batch_size, lr_schedule, backbone, checkpoint_after_iter, temp_path, device, seed, config_filename, nr_evaluations)

Constructor parameters:

- **env**: *gym.Env*\
  Reinforcement learning environment to train or evaluate the agent on.
- **lr**: *float, default=1e-5*\
  Specifies the initial learning rate to be used during training.
- **ent_coef**: *float, default=0.005*\
  Specifies the entropy coefficient used as additional loss penalty during training.
- **clip_range**: *float, default=0.1*\
  Specifies the clipping parameter for PPO.
- **gamma**: *float, default=0.99*\
  Specifies the discount factor during training.
- **n_steps**: *int, default=2048*\
  Specifies the number of steps to run for each environment per update during training.
- **n_epochs**: *int, default=4*\
  Specifies the number of epochs when optimizing the surrogate loss during training.
- **iters**: *int, default=6_000_000*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*\
  Specifies the batch size during training.
- **lr_schedule**: *{'', 'linear'}, default='linear'*\
  Specifies the learning rate scheduler to use. Empty to use a constant rate.
  Currently not implemented.
- **backbone**: *{'MultiInputPolicy'}, default='MultiInputPolicy'*\
  Specifies the architecture for the RL agent.
- **checkpoint_after_iter**: *int, default=20_000*\
  Specifies per how many training steps a checkpoint should be saved.
  If it is set to 0 no checkpoints will be saved.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm stores log files and saves checkpoints.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **seed**: *int, default=None*\
  Random seed for the agent.
  If None a random seed will be used.
- **config_filename**: *str, default=''*\
  Specifies the configuration file with important settings for the Simulator and PPO.
- **nr_evaluations**: *int, default=75*\
  Number of episodes to evaluate over.

#### `MultiObjectSearchRLLearner.fit`
```python
MultiObjectSearchRLLearner.fit(self, env, logging_path, silent, verbose)
```

Trains the agent on the environment.

Parameters:

- **env**: *gym.Env, default=None*\
  If specified use this env to train.
- **logging_path**: *str, default=''*\
  Path for logging and checkpointing.


#### `MultiObjectSearchRLLearner.eval`
```python
MultiObjectSearchRLLearner.eval(self, env, name_prefix, name_scene, nr_evaluations, deterministic_policy)
```
Evaluates the agent on the specified environment.

Parameters:

- **env**: *gym.Env*\
  Environment to evaluate on.
- **name_prefix**: *str, default=''*\
  Name prefix for all logged variables.
- **name_scene**: *str, default=''\
  Name of the iGibson scene.
- **nr_evaluations**: *int, default=75*\
  Number of episodes to evaluate over.
- **deterministic_policy**: *bool, default=False*\
  Use deterministic or stochastic policy.


#### `MultiObjectSearchRLLearner.save`
```python
MultiObjectSearchRLLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `MultiObjectSearchRLLearner.load`
```python
MultiObjectSearchRLLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.


  #### `MultiObjectSearchRLLearner.infer`
  ```python
  MultiObjectSearchRLLearner.infer(self, batch, deterministic)
  ```
  Loads a model from the path provided.

  Parameters:

  - **batch**: *int*\
    Number of samples to infer.
  - **deterministic**: *bool, default=False*\
    Use deterministic or stochastic policy.



#### Simulation Setup
The repository uses the iGibson Simulator as well as Stable-Baselines3 as external libraries.

This means that for the training environment to run, it relies on using iGibson scenes.
For that it is necessary to download the iGibson scenes.
A script is provided in [multi_object_search]
(/src/opendr/control/multi_object_search/requirements_installations.py)
To download he iGibson and the inflated traversability maps, please execute the following script and accept the agreement.

```sh
python requirements_installations.py
````

The iGibson dataset requires a valid license, which needs to be added manually.
The corresponding link can be found [here](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform).
In order to validate the iGibson dataset, copy the igibson.key file into the igibson/data/ folder.
For more information please have a look on the official website: https://stanfordvl.github.io/iGibson/dataset.html

##### Visualization
To visualize the egocentric maps and their corresponding static map, add the flag `show_map=true` in`config.yaml`.


#### Examples
* **Training and evaluation in the iGibson environment on a Multi Object Task**
As described above, follow the download instructions.
  ```python
    import torch
    from typing import Callable
    from opendr.control.multi_object_search import MultiObjectSearchRLLearner
    from opendr.control.multi_object_search import MultiObjectEnv
    from opendr.control.multi_object_search.algorithm.SB3.vec_env import VecEnvExt
    from pathlib import Path
    from igibson.utils.utils import parse_config


    def main():
        def make_env(rank: int, seed: int = 0, data_set=[]) -> Callable:

            def _init() -> MultiObjectEnv:
                env_ = MultiObjectEnv(
                    config_file=CONFIG_FILE,
                    scene_id=data_set[rank],
                    mix_sample=mix_sample[data_set[rank]]
                    )
                env_.seed(seed + rank)
                return env_
            return _init

        main_path = Path(__file__).parent
        logpath = f"{main_path}/logs/demo_run"
        CONFIG_FILE = str(f"{main_path}/best_defaults.yaml")

        mix_sample = {'Merom_0_int': False}
        train_set = ['Merom_0_int']

        env = VecEnvExt([make_env(0, data_set=train_set)])
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = parse_config(CONFIG_FILE)

        agent = MultiObjectSearchRLLearner(env, device=device, iters=config.get('train_iterations', 500),temp_path=logpath,config_filename=CONFIG_FILE)

        # start training
        agent.fit(env)

        # evaluate on finding 6 objects on one test scene
        scene = "Benevolence_1_int"
        metrics = agent.eval(env,name_prefix='Multi_Object_Search', name_scene=scene, nr_evaluations= 75,deterministic_policy = False)

        print(f"Success-rate for {scene} : {metrics['metrics']['success']} \nSPL for {scene} : {metrics['metrics']['spl']}")


    if __name__ == '__main__':
        main()
  ```

* **Evaluate a pretrained model**

  ```python
    import torch
    from opendr.control.multi_object_search import MultiObjectSearchRLLearner
    from opendr.control.multi_object_search import MultiObjectEnv
    from pathlib import Path
    from igibson.utils.utils import parse_config

    def main():
      main_path = Path(__file__).parent
      logpath = f"{main_path}/logs/demo_run"
      # best_defaults.yaml contains important settings. (see above)
      CONFIG_FILE = str(f"{main_path}/best_defaults.yaml")

      env = MultiObjectEnv(config_file=CONFIG_FILE, scene_id="Benevolence_1_int")

      device = "cuda" if torch.cuda.is_available() else "cpu"

      config = parse_config(CONFIG_FILE)

      agent = MultiObjectSearchRLLearner(env, device=device, iters=config.get('train_iterations', 500),temp_path=logpath,config_filename=CONFIG_FILE)

      # evaluate on finding 6 objects on all test scenes
      eval_scenes = ['Benevolence_1_int', 'Pomaria_2_int', 'Benevolence_2_int', 'Wainscott_0_int', 'Beechwood_0_int',
                'Pomaria_1_int', 'Merom_1_int']

      agent.load("pretrained")

      deterministic_policy = config.get('deterministic_policy', False)

      for scene in eval_scenes:
        metrics = agent.eval(env,name_prefix='Multi_Object_Search', name_scene=scene, nr_evaluations= 75,\
                  deterministic_policy = deterministic_policy)

        print(f"Success-rate for {scene} : {metrics['metrics']['success']} \nSPL for {scene} : {metrics['metrics']['spl']}")


    if __name__ == '__main__':
        main()
  ```

#### Notes

The iGibson simulator might crash, when evaluating multiple environments while using the gui mode (in .yaml file).

#### References
<a name="multi-object-search" href="https://arxiv.org/abs/2205.11384">[1]</a> Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces,
[arXiv](https://arxiv.org/abs/2205.11384).
