import pandas as pd
import pooch

def download_supp_data(fname="Oak2020_supp_data.xlsx"):
    """
    Download the supplementary data from the Oak 2020 paper.
    Source: https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-020-00744-3#Sec20
    """
    return pooch.retrieve("https://static-content.springer.com/esm/art%3A10.1186%2Fs13073-020-00744-3/MediaObjects/13073_2020_744_MOESM2_ESM.xlsx",
                          known_hash="17d388786d92d3dd82e7baefb993e24620787bdeb42f5cf90cc7ce0912556ab5",
                          fname=fname)

 
def read_supp_data(sheet_name=None, 
                   **kwargs):
    
    """
    Read the supplementary data from the Oak 2020 paper.
    Source: https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-020-00744-3#Sec20
    
    Args:
        sheet_name: Name of the sheet to read. If None, the first sheet is used.
        **kwargs: Additional arguments to pass to pd.read_excel.

    Returns:
        pd.DataFrame: The data from the sheet.

    Example:
        s4a = read_supp_data(sheet_name='S4a.PredisposingVariants')
        s4a.head() 

        s4b = read_supp_data(sheet_name='S4b.SomaticSecondHit') 
        s4b.head()   
    """
    path =  download_supp_data()
    sheet_names = pd.ExcelFile(path).sheet_names
    if sheet_name is None:
        print(f"No sheet name provided. Using first sheet: '{sheet_names[0]}'")
        sheet_name = sheet_names[0]
    else:
        if sheet_name not in sheet_names:
            raise ValueError(f"Sheet name {sheet_name} not found in {path}. Available sheets: {sheet_names}")
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)