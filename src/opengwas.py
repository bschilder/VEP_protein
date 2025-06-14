import ieugwaspy
import os 
import pandas as pd



def get_opengwas_info(cache_path = os.path.join(os.path.expanduser("~"), ".cache", "ieugwaspy.parquet")):
    if not os.path.exists(cache_path):
        ginfo = ieugwaspy.gwasinfo()
        opengwas = pd.DataFrame(ginfo).T
        opengwas.to_parquet(cache_path, index=False)
    else:
        opengwas = pd.read_parquet(cache_path)
    return opengwas