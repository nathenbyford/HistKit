# HistKit

HistKit is a Python library for working with histogram data. It provides a `Histogram` and `HistogramMatrix` dataclasses that makes it easy to create, manipulate, and analyze histograms.

## Features

- Create histograms from raw data or from breaks and counts.
- Calculate histogram properties like mean, standard deviation, and probabilities.
- Pretty print histograms in a table format.
- Visualize histograms using `matplotlib`.


## Installation

You can install HistKit directly from GitHub using pip:

```bash
pip install git+https://github.com/nathenbyford/HistKit.git
```

## Usage

Here is a simple example of how to use HistKit:

```python
import numpy as np
from histkit.hist import Histogram

# Create a histogram from raw data
data = np.random.normal(size=1000)
hist = Histogram.from_raw_data(data, bins=10)

# Print the histogram
print(hist)

# Get histogram properties
print(f"Mean: {hist.mean}")
print(f"Standard deviation: {hist.std}")

# Plot the histogram
hist.plot(color='blue', edgecolor='black')
```

This will output:

```
          X                   p      
Bin_1     [ -3.06 ; -2.48 )          0.011   
Bin_2     [ -2.48 ; -1.9 )           0.033   
Bin_3     [ -1.9 ; -1.32 )           0.093   
Bin_4     [ -1.32 ; -0.742 )         0.16    
Bin_5     [ -0.742 ; -0.163 )        0.201   
Bin_6     [ -0.163 ; 0.415 )         0.222   
Bin_7     [ 0.415 ; 0.994 )          0.168   
Bin_8     [ 0.994 ; 1.57 )           0.081   
Bin_9     [ 1.57 ; 2.15 )            0.024   
Bin_10    [ 2.15 ; 2.73 ]            0.007   

mean = -0.163...         std = 0.988...
Mean: -0.163...
Standard deviation: 0.988...
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
