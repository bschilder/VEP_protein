PARAMS_VARIATION = {
    'genotypes': 1,
    # 'genotyping_chips': 1,
    'phenotypes': 1,
    'pops': 1,
    'population_genotypes':1
}

## VEP
PARAMS_VEP = {
    'AlphaMissense': 1,
    'CADD': 1,
    'ClinPred': 1,
    'EVE': 1,
    'Enformer': 1,
    'MaveDB': 1,
    'REVEL': 1,
    'pick': 1
}
    
def set_params_vep(params):
    global PARAMS_VEP
    PARAMS_VEP.update(params)
    print("PARAMS_VEP has been updated.")

def get_params_vep():
    return PARAMS_VEP.copy() 

## HAPLOTYPES
PARAMS_HAPLOTYPES = {
    'samples': 1,
    'sequence': 1,
    'aligned_sequences': 1
}

def set_params_haplotypes(params):
    global PARAMS_HAPLOTYPES
    PARAMS_HAPLOTYPES.update(params)
    print("PARAMS_HAPLOTYPES has been updated.")

def get_params_haplotypes():
    return PARAMS_HAPLOTYPES.copy() 

## PALETTES
PALETTES = {
    'WT': 'Blues',
    'Pathogenic': 'Oranges',
    'Benign': 'Greens'
}

def set_params_palettes(params):
    global PARAMS_PALETTES
    PARAMS_PALETTES.update(params)
    print("PARAMS_PALETTES has been updated.")

def get_params_palettes():
    return PARAMS_PALETTES.copy() 

