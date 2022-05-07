import pandas as pd
from typing import Optional, Iterable, Tuple, List
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64
import hashlib
import aiohttp
import asyncio


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
            self.sub_dir = subdir
            urls, ids = batch_df['url'].to_list(), batch_df['id'].to_list()
            paths = list(map(self.generate_img_path, ids))
            local_csv_path = os.path.join(self.dfs_folder, f'{idx}.parquet.gzip')
            batch_df['local_paths'] = paths
            batch_df.to_parquet(local_csv_path, compression='gzip')
            idx += 1
            asyncio.run(self._load_batch(urls, paths))

    async def _load_batch(self, urls: List[str], paths: List[Path]) -> None:
        local_paths = map(self.file_download_task, urls, paths)
        await asyncio.wait(local_paths)

    async def file_download_task(self, url: str, path: Path):
        content = await self._load_sample(url)
        local_path = await self._save_img(content, path)
        return local_path

    async def _save_img(self, content: bytes, local_path: Path) -> Optional[str]:
        if content:
            try:
                image = Image.open(io.BytesIO(content)).convert("RGB")
                image = image.resize(self.img_size)
                image.save(fp=local_path, quality=self.quality, format='JPEG')
                return local_path
            except:
                return None

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

    @staticmethod
    async def _load_sample(url: str) -> Optional[str]:
        content = None
        if url.startswith('ipfs://'):
            url = url.replace('ipfs://', 'https://ipfs.io/ipfs/')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    content = await response.read()
        except:
            pass
        finally:
            return content

