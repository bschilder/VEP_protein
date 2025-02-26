# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.gprofiler as gp
import src.proteingym as pg


# %%
proteins_df = pg.merge_resources()
proteins_df

# %%
id_map = gp.get_id_map(ids=proteins_df['protein'].unique().tolist(), 
                       drop_na=None,
                       target_namespace='ENSP')
print(sum(id_map['incoming']=='None'))
print(sum(id_map['converted']=='None'))
id_map.n_incoming.describe()
id_map.n_converted.describe()
id_map.namespaces.value_counts()
# %%
