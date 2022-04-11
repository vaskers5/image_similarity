import pandas as pd
import requests
from typing import Optional
from pathlib import Path
import os
from tqdm import tqdm
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True)


class CloudImgDatasetLoader:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._dataframe_preprocessing()
        self.folder = None
        self.img_folder = None
        
    def __call__(self, out_dataset_path: str) -> None:
        self.folder = os.path.abspath(out_dataset_path)
        self.img_folder = os.path.join(self.folder, 'img_data')
        if not(os.path.exists(self.img_folder)):
            os.mkdir(self.img_folder)
        self.df['local_path'] = self.df.parallel_apply(lambda x: self._load_data(x.id, x.url), axis=1)
        return self.df
        
        
    def _dataframe_preprocessing(self) -> None:
        self.df = self.df.dropna(subset=['id', 'url'])
        self.df = self.df.drop_duplicates(subset=['id', 'url'], keep='first', ignore_index=True)
    
    
    def _load_data(self, img_id: str, url: str) -> Optional[str]:
        if url.startswith('ipfs://'):
            url = url.replace('ipfs://', 'https://ipfs.io/ipfs/')
        try:
            data = requests.get(url, timeout=10).content
        except:
            return None
        local_path = os.path.join(self.img_folder, f'{img_id}.png')
        with open(local_path, 'wb') as f:
            f.write(data)
        return local_path
