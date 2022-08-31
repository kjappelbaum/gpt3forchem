# Data used in this study 

- `polymers.csv`: Data used for the polymer case study. Taken from [10.1038/s41467-021-22437-0](https://www.nature.com/articles/s41467-021-22437-0). We added the categorical column using `pd.cut`
- `fragprint_features.csv`: "Fragprint" feature vectors for the photoswitch dataset, [computed using the original code](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/property_prediction/data_utils.py).
- `photoswitches.csv`: [Photoswitch dataset](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset) with additional `name` (IUPAC name, as retrieved from [the chemical name resolver](https://cactus.nci.nih.gov/chemical/structure)) and [`selfies`](https://github.com/aspuru-guzik-group/selfies) column.