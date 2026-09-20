# Data

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset
from the UCI Machine Learning Repository (dataset ID 296).

- Official record: https://archive.ics.uci.edu/dataset/296/
- Original study: Strack et al. (2014), DOI 10.1155/2014/781670
- UCI license shown for this dataset: CC BY 4.0

The official dataset can be downloaded using:

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=296)
frame = pd.concat(
    [dataset.data.features, dataset.data.targets],
    axis=1
)

frame.to_csv("data/diabetic_data.csv", index=False)
