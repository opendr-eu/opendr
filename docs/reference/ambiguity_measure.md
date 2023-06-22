## ambiguity_measure module

The *ambiguity_measure* module contains the *AmbiguityMeasure* class.

### Class AmbiguityMeasure
Bases: `object`

The *AmbiguityMeasure* class is a tool that allows to obtain an ambiguity measure of vision-based models that output pixel-wise value estimates.
This tool can be used in combination with vision-based manipulation models such as Transporter Nets [[1]](#transporter-paper).

The [AmbiguityMeasure](../../src/opendr/utils/ambiguity_measure/ambiguity_measure.py) class has the following public methods:

#### `AmbiguityMeasure` constructor
```python
AmbiguityMeasure(self, threshold, temperature)
```

Constructor parameters:

- **threshold**: *float, default=0.5*\
  Ambiguity threshold, should be in [0, 1).
- **temperature**: *float, default=1.0*\
  Temperature of the sigmoid function.
  Should be > 0.
  Higher temperatures will result in higher ambiguity measures.

#### `AmbiguityMeasure.get_ambiguity_measure`
```python
AmbiguityMeasure.get_ambiguity_measure(self, heatmap)
```

This method allows to obtain an ambiguity measure of the output of a model.

Parameters:

- **heatmap**: *np.ndarray*\
  Pixel-wise value estimates.
  These can be obtained using from for example a Transporter Nets model [[1]](#transporter-paper).

#### Demos and tutorial

A demo showcasing the usage and functionality of the *AmbiguityMeasure* is available [here](https://colab.research.google.com/github/opendr-eu/opendr/blob/ambiguity_measure/projects/python/utils/ambiguity_measure/ambiguity_measure_tutorial.ipynb).


#### Examples

* **Ambiguity measure example**

  This example shows how to obtain the ambiguity measure from pixel-wise value estimates.

  ```python
  import numpy as np
  from opendr.utils.ambiguity_measure.ambiguity_measure import AmbiguityMeasure
  
  # Simulate image and value pixel-wise value estimates (normally you would get this from a model such as Transporter)
  img = 255 * np.random.random((128, 128, 3))
  img = np.asarray(img, dtype="uint8")
  heatmap = 10 * np.random.random((128, 128))
  
  # Initialize ambiguity measure
  am = AmbiguityMeasure(threshold=0.1, temperature=3)
  
  # Get ambiguity measure of the heatmap
  ambiguous, locs, maxima, probs = am.get_ambiguity_measure(heatmap)
  
  # Plot ambiguity measure
  am.plot_ambiguity_measure(heatmap, locs, probs, img)
  ```

#### References
<a name="transporter-paper" href="https://proceedings.mlr.press/v155/zeng21a/zeng21a.pdf">[1]</a>
Zeng, A., Florence, P., Tompson, J., Welker, S., Chien, J., Attarian, M., ... & Lee, J. (2021, October).
Transporter networks: Rearranging the visual world for robotic manipulation.
In Conference on Robot Learning (pp. 726-747).
PMLR.

