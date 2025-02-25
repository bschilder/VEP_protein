# VEP_protein
Using protein sequence models to compute Variant Effect Predictions (VEP) across biobank-scale populations.

## Directory description
- *conda/* : Files to define Conda environments.
- *notebooks/* : Jupyter notebooks with step-by-step examples.  
- *src/* : Source code scripts (e.g. `utils.py`) to be called by notebooks.


## Getting started

- Clone this repo to your machine and open it as a project in your code editor (e.g. VSCode), or simply `cd` into the folder from the Terminal.  
- Adjust `DATA_DIR` variable in the [`config.py` file](https://github.com/bschilder/VEP_protein/blob/main/src/config.py) to the on-disk location where you'd like to store files.  
- Follow the [VEP tutorial notebook](https://github.com/bschilder/VEP_protein/blob/main/notebooks/VEP.ipynb).