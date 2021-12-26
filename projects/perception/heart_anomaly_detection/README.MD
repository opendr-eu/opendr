# Heart Anomaly Detection

This folder contains a sample application for heart anomaly detection.
It can be used for both evaluation on AF dataset, as well as inference on a single ECG timeseries provided as .npy array.
Demo for evaluation on AF dataset can be run as follows:

```python
python3 demo.py -model [anbof|gru] 
```
By setting ` --input_data /path/to/data.npy` inference can be performed from a single ECG sequence. 

Additional parameters can be specified to control the model and data parameters:

- ` --attention_type` sets the attention type for ANBOF model
- ` --checkpoint` defines the path to a pretrained model (AF-pretrained temporal ANBOF or GRU will be downloaded by default, otherwise a path to existing checkpoint should be specified)
- ` --channels` defines the number of channels in input sequence
- ` --series_length` defines the length of input sequence
- ` --n_class` defines the number of classes

If you switch between the two available models (ANBOF and GRU) without specifying the checkpoint folder, please make sure that you delete the previous model files (e.g., `rf -rf checkpoint`).
 
