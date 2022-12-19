## Continual Transformer Encoder module


### Class CoTransEncLearner
Bases: `engine.learners.Learner`

The *CoTransEncLearner* class provides a Continual Transformer Encoder learner, which can be used for time-series processing of user-provided features.
This module was originally proposed by Hedegaard et al. in "Continual Transformers: Redundancy-Free Attention for Online Inference", 2022, https://arxiv.org/abs/2201.06268"

The [CoTransEncLearner](src/opendr/perception/activity_recognition/continual_transformer_decoder/continual_transformer_decoder_learner.py) class has the following public methods:

#### `CoTransEncLearner` constructor

```python
CoX3DLearner(self, lr, iters, batch_size, optimizer, lr_schedule, network_head, num_layers, input_dims, hidden_dims, sequence_len, num_heads, dropout, num_classes, positional_encoding_learned, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, loss, weight_decay, momentum, drop_last, pin_memory, num_workers, seed)
```

Constructor parameters:

  - **lr**: *float, default=1e-2*\
    Learning rate during optimization.
  - **iters**: *int, default=10*\
    Number of epochs to train for.
  - **batch_size**: *int, default=64*\
    Dataloader batch size. Defaults to 64.
  - **optimizer**: *str, default="sgd"*\
    Name of optimizer to use ("sgd" or "adam").
  - **lr_schedule**: *str, default=""*\
    Schedule for training the model.
  - **network_head**: *str, default="classification"*\
    Head of network (only "classification" is currently available).
  - **num_layers**: *int, default=1*\
    Number of Transformer Encoder layers (1 or 2). Defaults to 1.
  - **input_dims**: *float, default=1024*\
    Input dimensions per token.
  - **hidden_dims**: *float, default=1024*\
    Hidden projection dimension.
  - **sequence_len**: *int, default=64*\
    Length of token sequence to consider.
  - **num_heads**: *int, default=8*\
    Number of attention heads.
  - **dropout**: *float, default=0.1*\
    Dropout probability.
  - **num_classes**: *int, default=22*\
    Number of classes to predict among.
  - **positional_encoding_learned**: *bool, default=False*\
    Positional encoding type.
  - **checkpoint_after_iter**: *int, default=0*\
    Unused parameter.
  - **checkpoint_load_iter**: *int, default=0*\
    Unused parameter.
  - **temp_path**: *str, default=""*\
    Path in which to store temporary files.
  - **device**: *str, default="cuda"*\
    Name of computational device ("cpu" or "cuda").
  - **loss**: *str, default="cross_entropy"*\
    Loss function used during optimization.
  - **weight_decay**: *[type], default=1e-4*\
    Weight decay used for optimization.
  - **momentum**: *float, default=0.9*\
    Momentum used for optimization.
  - **drop_last**: *bool, default=True*\
    Drop last data point if a batch cannot be filled.
  - **pin_memory**: *bool, default=False*\
    Pin memory in dataloader.
  - **num_workers**: *int, default=0*\
    Number of workers in dataloader.
  - **seed**: *int, default=123*\
    Random seed.


#### `CoTransEncLearner.fit`
```python
CoTransEncLearner.fit(self, dataset, val_dataset, epochs, steps)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:
  - **dataset**: *Dataset*:
    Training dataset.
  - **val_dataset**: *Dataset, default=None*
    Validation dataset. If none is given, validation steps are skipped.
  - **epochs**: *int, default=None*
    Number of epochs. If none is supplied, self.iters will be used.
  - **steps**: *int, default=None*
    Number of training steps to conduct. If none, this is determined by epochs.


#### `CoTransEncLearner.eval`
```python
CoTransEncLearner.eval(self, dataset, steps)
```
This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:
  - **dataset**: *Dataset*
    Dataset on which to evaluate model.
  - **steps**: *int, default=None*
    Number of validation batches to evaluate. If None, all batches are evaluated.


#### `CoTransEncLearner.infer`
```python
CoTransEncLearner.infer(x)
```

This method is used to perform classification of a video.
Returns a `engine.target.Category` objects, where each holds a category.

Parameters:
- **x**: *Union[Timeseries, Vector, torch.Tensor]*
  Either a single time instance (Vector) or a Timeseries. x can also be passed as a torch.Tensor.


#### `CoTransEncLearner.save`
```python
CoTransEncLearner.save(self, path)
```

Save model weights and metadata to path.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as "model_name.pth" and the metadata file as "name.json".
If the files already exist, their names are versioned with a suffix.

If `self.optimize` was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension.

Parameters:
- **path**: *str*
  Directory in which to save model weights and meta data.


#### `CoTransEncLearner.load`
```python
CoTransEncLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.

Parameters:
- **path**: *str*
  Path to metadata file in json format or to weights path.


#### `CoTransEncLearner.optimize`
```python
CoTransEncLearner.optimize(self, do_constant_folding)
```

Optimize model execution. This is accomplished by saving to the ONNX format and loading the optimized model.

Parameters:
- **do_constant_folding**: *bool, default=False*
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.


#### Examples

* **Fit model**.

  ```python
  from opendr.perception.activity_recognition import CoTransEncLearner
  from opendr.perception.activity_recognition.datasets import DummyTimeseriesDataset

  learner = CoTransEncLearner(
      batch_size=2,
      device="cpu",
      input_dims=8,
      hidden_dims=32,
      sequence_len=64,
      num_heads=8,
      num_classes=4,
  )
  train_ds = DummyTimeseriesDataset(
      sequence_len=64, num_sines=8, num_datapoints=128
  )
  val_ds = DummyTimeseriesDataset(
      sequence_len=64, num_sines=8, num_datapoints=128, base_offset=128
  )
  learner.fit(dataset=train_ds, val_dataset=val_ds, steps=2)
  learner.save('./saved_models/trained_model')
  ```

* **Evaluate model**.

  ```python
  from opendr.perception.activity_recognition import CoTransEncLearner
  from opendr.perception.activity_recognition.datasets import DummyTimeseriesDataset

  learner = CoTransEncLearner(
      batch_size=2,
      device="cpu",
      input_dims=8,
      hidden_dims=32,
      sequence_len=64,
      num_heads=8,
      num_classes=4,
  )
  test_ds = DummyTimeseriesDataset(
      sequence_len=64, num_sines=8, num_datapoints=128, base_offset=256
  )
  results = learner.eval(test_ds)  # Dict with accuracy and loss
  ```


#### References
<a name="cotransenc" href="https://arxiv.org/abs/2201.06268">[3]</a> Continual Transformers: Redundancy-Free Attention for Online Inference,
[arXiv](https://arxiv.org/abs/2201.06268).
