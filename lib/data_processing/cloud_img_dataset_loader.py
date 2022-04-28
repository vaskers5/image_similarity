import pandas as pd
import requests
from typing import Optional, Iterable, Tuple
import os
from tqdm import tqdm
import concurrent.futures
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64
import hashlib


class CloudImgDatasetLoader:
    def __init__(self,
                 df: pd.DataFrame,
                 num_splits: int,
                 saved_img_quality: int = 80,
                 img_size: Tuple[int, int] = (256, 256)):
        r"""

        Class for dataset downloading use it when you want download a very large data,
        node that your dataset need to contain image urls(url - column),
        image id(id - column)

        Parameters
        ----------

        df : pd.Dataframe object

        num_splits : int num of dataset checkpoints, where len of one
            checkpoint is len(df) / num_splits

        saved_img_quality: int quality of saved images

        img_size: Tuple(int, int) default is (256, 256) the size of saved images


        """
        self.df_loader = self.split_df_iterator(self._dataframe_preprocessing(df), num_splits)
        self.folder = None
        self.img_folder = None
        self.dfs_folder = None
        self.quality = saved_img_quality
        self.img_size = img_size

    def __call__(self, out_dataset_path: str) -> None:
        r""" Function for start downloading.

        File structure after downloading:

        -image_data: directory with folders of images

        -datasets: directory with checkpoints of dataframes saved in csv format

        Parameters
        ----------

        out_dataset_path : str path for out dataset folder

        """
        self._make_main_dirs(out_dataset_path)
        self._load_full_data()
        self._save_result_df()

    def _make_main_dirs(self, out_dataset_path: str) -> None:
        self.folder = os.path.abspath(out_dataset_path)
        self.img_folder = os.path.join(self.folder, 'img_data')
        self.dfs_folder = os.path.join(self.folder, 'datasets')
        self._make_dir(self.img_folder), self._make_dir(self.dfs_folder)

    def _load_full_data(self) -> None:
        idx = 0
        for batch_df in tqdm(self.df_loader):
            subdir = os.path.join(self.img_folder, str(idx))
            self._make_dir(subdir)
            batch_df = self._load_batch(subdir, batch_df)
            local_csv_path = os.path.join(self.dfs_folder, f'{idx}.parquet.gzip')
            batch_df.to_parquet(local_csv_path, compression='gzip')
            idx += 1

    def _load_batch(self, subdir: Path, batch_df: pd.DataFrame) -> pd.DataFrame:
        urls, ids = batch_df['url'].to_list(), batch_df['id'].to_list()
        self.sub_dir = subdir
        paths = list(map(self.generate_img_path, ids))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            local_paths = list(tqdm(executor.map(self._load_sample, paths, urls),
                                    leave=False,
                                    total=len(urls)))

        batch_df['local_paths'] = local_paths
        return batch_df

    def _load_sample(self, local_path: Path, url: str) -> Optional[str]:
        if url.startswith('ipfs://'):
            url = url.replace('ipfs://', 'https://ipfs.io/ipfs/')
        try:
            data = requests.get(url, timeout=10).content
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except:
            return None
        image = image.resize(self.img_size)
        image.save(fp=local_path, quality=self.quality, format='JPEG')
        return local_path

    def generate_img_path(self, img_id: str) -> Path:
        hasher = hashlib.sha1(str(img_id).encode('utf-8'))
        image_name = str(base64.urlsafe_b64encode(hasher.digest()).decode('utf-8')).replace('=', '')
        n = 2
        sub_dirs = [image_name[i:i + n] for i in range(0, len(image_name), n)][:3]
        res_path = self.sub_dir
        for sub_dir_img in sub_dirs:
            res_path = os.path.join(res_path, sub_dir_img)
        self._make_dir(res_path)
        res_path = os.path.join(res_path, f'{image_name}.jpg')
        return res_path

    def _save_result_df(self) -> None:
        dfs = [pd.read_parquet(os.path.join(self.dfs_folder,
                                            path)) for path in os.listdir(self.dfs_folder)]
        res_path = os.path.join(self.dfs_folder, 'full_data.parquet.gzip')
        pd.concat(dfs, ignore_index=True).to_parquet(res_path, compression='gzip')

    @staticmethod
    def _dataframe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['id', 'url'])
        df = df.drop_duplicates(subset=['id', 'url'], keep='first', ignore_index=True)
        return df

    @staticmethod
    def split_df_iterator(df: pd.DataFrame, chunks: int) -> Iterable[pd.DataFrame]:
        for chunk in np.array_split(df, chunks):
            yield chunk

    @staticmethod
    def _make_dir(dir_path: str) -> None:
        if not(os.path.exists(dir_path)):
            os.makedirs(dir_path)

