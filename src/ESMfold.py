import os
from tqdm import tqdm
from time import time

def install(version = "1"):
    #@title install
    #@markdown install ESMFold, OpenFold and download Params (~2min 30s)
    
    model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
    if not os.path.isfile(model_name):
        # download esmfold params
        os.system("apt-get install aria2 -qq")
        os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &")

    if not os.path.isfile("finished_install"):
        # install libs
        print("installing libs...")
        os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif")
        os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")

        print("installing openfold...")
        # install openfold
        os.system(f"pip install -q git+https://github.com/sokrypton/openfold.git")

        print("installing esmfold...")
        # install esmfold
        os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")
        os.system("touch finished_install")

    # wait for Params to finish downloading...
    while not os.path.isfile(model_name):
        time.sleep(5)
    if os.path.isfile(f"{model_name}.aria2"):
        print("downloading params...")
    while os.path.isfile(f"{model_name}.aria2"):
        time.sleep(5)