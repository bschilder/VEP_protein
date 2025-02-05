import pooch

def get_resources_df(version = 'v1.1',
                     save_path = 'ProteinGym/ProteinGym_data_urls.tsv',
                     force=False):
    # Copied from here: https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#resources
    file_text = """
    Data	Size (unzipped)	Filename	Raw	Hash
    DMS benchmark - Substitutions	1.0GB	DMS_ProteinGym_substitutions.zip	False	3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921
    DMS benchmark - Indels	200MB	DMS_ProteinGym_indels.zip	False	5c5c7446a8c8f89534dfa87e546d2f9c00590d19aa5ce4c01d271abc7c962f74
    Zero-shot DMS Model scores - Substitutions	31GB	zero_shot_substitutions_scores.zip	False	22df5c0f47e8278b39d0c1a51518e20d674b5109e136578bbede660af2bd7ecd
    Zero-shot DMS Model scores - Indels	5.2GB	zero_shot_indels_scores.zip	False	957dc5d0d3e4163f56b3d45b865150a44fcd8ea9e2cf172e9c3fbbac2e344d81
    Supervised DMS Model performance - Substitutions	2.7MB	DMS_supervised_substitutions_scores.zip	False	8167ff7eee01e748a7820034940847f888532cb2c942bc9ae18e413f77bce2cb
    Supervised DMS Model performance - Indels	0.9MB	DMS_supervised_indels_scores.zip	False	3cf375bc9ae80b878e6c55ddeade2ef5f2895d479e4d414872d205007351bf15
    Multiple Sequence Alignments (MSAs) for DMS assays	5.2GB	DMS_msa_files.zip	False	f8c894f0f113f5f49f2945c512b73f488bdf582097dff04658fbb703d92fe34d
    Redundancy-based sequence weights for DMS assays	200MB	DMS_msa_weights.zip	False	2f36a2a7882b264142eca273255da659fc8640249234edf934ffef364a585084
    Predicted 3D structures from inverse-folding models	84MB	ProteinGym_AF2_structures.zip	False	c78f5ff60cf59104fe19b8318c5647587aad033ee832e051d0efec8e137c423a
    Clinical benchmark - Substitutions	123MB	clinical_ProteinGym_substitutions.zip	False	afe711af49365bc1ee220a5d212c570a4d9bc35e6960d19a93a0d1ed4ce37be4
    Clinical benchmark - Indels	2.8MB	clinical_ProteinGym_indels.zip	False	644192ef474998346ff760c3b3d6d0d731aebf79ce3c5057e3f2748c687128d6
    Clinical MSAs	17.8GB	clinical_msa_files.zip	False	9f55b0792419f0f7f0d64f39f5345bb1510db5e02fb7a85347db3b0d2f8b3531
    Clinical MSA weights	250MB	clinical_msa_weights.zip	False	564bbef2a6f22e544fc88ea49a31f1d1e585ad663e17d4d1e5f78f06a412fa49
    Clinical Model scores - Substitutions	0.9GB	zero_shot_clinical_substitutions_scores.zip	False
    Clinical Model scores - Indels	0.7GB	zero_shot_clinical_indels_scores.zip	False	1834dfe2a43e34529eea77c1dbe7b0503153578455b7b146856b31268ee17aa7
    CV folds - Substitutions - Singles	50M	cv_folds_singles_substitutions.zip	False	920f0be936233b96b5052cd23679e42355cfd2b4e6f45b4f571eb79c0b2f9c35
    CV folds - Substitutions - Multiples	81M	cv_folds_multiples_substitutions.zip	False	4f1453ee8ccf2d38f23ae43f97fc7f962e54e5f10390711b59f6929538dd25f9
    CV folds - Indels	19MB	cv_folds_indels.zip	False	b3f123321b499b470da03ddd3530241502851152f9a98775ecd6b508ae9c856d
    DMS benchmark: Substitutions (raw)	500MB	substitutions_raw_DMS.zip	True	6d83b16585de2b71b67ae1985193b9eec2e01804784286c515ff276b5372e412
    DMS benchmark: Indels (raw)	450MB	indels_raw_DMS.zip	True	93c21d4cdc09755428e417e330fdf7b3bf16705f125b23df208648b3ca5595a0
    Clinical benchmark: Substitutions (raw)	58MB	substitutions_raw_clinical.zip	True	caa461bd2e0c58501131e7c1ad9d26c118c67704efe1b67c7ff7ca1d72ae7275
    Clinical benchmark: Indels (raw)	12.4MB	indels_raw_clinical.zip	True	f9eb7232657ab5732eda8dcb922bf17b228eae212ca794e753ba73a017f40a8d
    """
    import pandas as pd
    import os
    if os.path.exists(save_path) and not force:
        from io import StringIO
        df = pd.read_csv(StringIO(file_text), sep='\t') 
        df['URL'] = df['Filename'].apply(lambda x: f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{version}/{x}")
        df['Version'] = version
        df.to_csv(save_path, index=False, sep='\t')
    else:
        df = pd.read_csv(save_path, sep='\t')
    return df

def download_resources(resources_df=get_resources_df(),
                       path=pooch.os_cache('ProteinGym'),
                       include_raw=False, 
                       remove_zip=False, 
                       error=True,
                       progressbar=1): 
    import os
    import pandas as pd
    from tqdm.auto import tqdm
    if not include_raw:
        resources_df = resources_df[~resources_df['Raw']]
        
    def _rm_zip(row, fpath):
        zipped_name = os.path.join(fpath, os.path.basename(row['URL'])+".zip")
        if os.path.exists(zipped_name):
            print(f"Removing compressed file: {zipped_name}")
            os.remove(zipped_name)

    file_dict = {}
    for i, row in tqdm(resources_df.iterrows(), 
                       total=len(resources_df), 
                       desc='Downloading resources', disable=progressbar<1): 
        try:
            fpath = os.path.join(path, row['Version'])
            unzipped_name = os.path.basename(row['URL']).removesuffix('.zip')
            processor = pooch.Unzip(extract_dir=unzipped_name)
            if row['Filename'].endswith('.zip') and remove_zip:
                if os.path.exists(unzipped_name) and remove_zip:
                    file_dict[unzipped_name] = []
                    for root, dirs, files in os.walk(unzipped_name):
                        for file in files:
                            file_dict[unzipped_name].append(os.path.join(root, file))
                    print(f"Skipping {unzipped_name} because it already exists")
                    if remove_zip:
                        _rm_zip(row, fpath)
                    continue
            file_dict[unzipped_name] = pooch.retrieve(url=row['URL'], 
                                                        fname=os.path.basename(row['URL']),
                                                        known_hash=None if pd.isna(row['Hash']) else row['Hash'], 
                                                        path=fpath,
                                                        progressbar=progressbar>1,
                                                        processor=processor) 
            if remove_zip:
                _rm_zip(row, fpath)
        except Exception as e:
            if error:
                raise e
            else:
                print(f"Error downloading {row['Filename']}: {e}")
    return file_dict

# Read and concatenate all clinical substitution files
def concat_csvs(pg_resources,
                fkey='clinical_ProteinGym_substitutions'):
    from tqdm.auto import tqdm
    clinical_subs_dfs = []
    for csv_path in tqdm(pg_resources[fkey],
                        desc="Reading files in " + fkey):
        df = pd.read_csv(csv_path, index_col=0)
        # Add source file name as column
        df['source_file'] = os.path.basename(csv_path)
        clinical_subs_dfs.append(df)
    return pd.concat(clinical_subs_dfs, ignore_index=True)