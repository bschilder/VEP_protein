# VEP_protein
Using protein sequence models to compute Variant Effect Predictions (VEP) across biobank-scale populations.

## Directory description
- *conda/* : Files to define Conda environments.
- *notebooks/* : Jupyter notebooks with step-by-step examples.  
- *src/* : Source code scripts (e.g. `utils.py`) to be called by notebooks.


## Getting started

1. Clone this repo to your machine and open it as a project in your code editor (e.g. VSCode), or simply `cd` into the folder from the Terminal.  
`git clone https://github.com/bschilder/VEP_protein.git && cd VEP_protein`

2.[Optional] Construct and activate the conda environment which should include all necessary dependencies:
`conda env create -f conda/conda.yml`

3. Adjust `DATA_DIR` variable in the [`config.py` file](https://github.com/bschilder/VEP_protein/blob/main/src/config.py) to the on-disk location where you'd like to store files.  

4. Follow the [VEP tutorial notebook](https://github.com/bschilder/VEP_protein/blob/main/notebooks/VEP.ipynb).
    - [Optional] If you followed Step 2 and created the conda env, select `vep_protein` as the notebook's kernel.


