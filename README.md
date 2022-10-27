## Install packages

### Install conda

*Recommended*: Install miniconda from Anaconda website [here](https://docs.conda.io/en/latest/miniconda.html).

### Create environment from YAML

```
conda env create -f environment.yml
```

### Activate the environment

`conda activate windengie`

### Edit and run the main

1. Place open turbine data from Engie in `data/` directory in the root directory. Create `fig/` directory for results.
2. Select the turbine for which a model has to be trained and tested. (0 to 4)
3. Models are built, trained and tested. Results are plotted in an HTML file and saved in `.fig/` directory.