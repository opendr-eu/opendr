# Speech Command Recognition

This folder contains a demo script for performing the speech command recognition on a given audio file, using one of the three currently available models: MatchboxNet, EdgeSpeechNets or Quadratic SelfONN. The script uses a pretrained model that can be either provided from a local file or downloaded from the OpenDR FTP server (the latter option is currently not available for EdgeSpeechNets).

The demo can be run as follows:

```python
python demo.py INPUT_FILE --model [matchboxnet|edgespeechnets|quad_selfonn] 
```

The following additional parameters are supported:

` --model_path` gives the path to a pretrained model, if not given, downloading from the OpenDR FTP server will be attempted

` --n_class` defines the number of classes supported by the model (default 20)
 
Two example recordings are also provided in the folder.
