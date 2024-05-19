import pandas as pd
from typing import Optional
import re

def read_and_rename(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except(FileNotFoundError):
        print("File does not exists!")
        return None
    
    #basic columns renaming
    rename_mapper = {k:re.sub("[^A-Z|_]", "", k.lower().replace(" ", "_") ,0,re.IGNORECASE) for k in df.columns}
    return df.rename(columns=rename_mapper)
    
    