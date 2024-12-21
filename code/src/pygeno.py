from pyGeno.Genome import Genome
from pyGeno.Gene import Gene
from pyGeno.Transcript import Transcript
from pyGeno.Protein import Protein

ref_path="/grid/koo/home/schilder/.local/lib/python3.9/site-packages/pyGeno/bootstrap_data/genomes/GRCh38/Human.GRCh38.109.tar.gz"
if not os.path.exists(ref_path):
    import pyGeno.bootstrap as B
    import os
    os.makedirs("/grid/koo/home/schilder/.local/lib/python3.9/site-packages/pyGeno/bootstrap_data/genomes/GRCh38/", exist_ok=True)
    !wget https://bioinfo.iric.ca/~feghalya/pyGeno_datawraps/Human.GRCh38.109.tar.gz -O {ref_path}
    B.importGenome("GRCh38/Human.GRCh38.109.tar.gz")



#the name of the genome is defined inside the package's manifest.ini file
ref = Genome(name = 'GRCh38.109') 

# gene = ref.get(Transcript)[0]
# for prot in gene.get(Protein):
#     print(prot.sequence)

# os.makedirs("/grid/koo/home/schilder/.local/lib/python3.9/site-packages/pyGeno/bootstrap_data/SNPs/", exist_ok=True)
# B.importSNPs("Human.dummySRY_casava.tar.gz")
