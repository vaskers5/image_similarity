import shutil

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
from loguru import logger
import magic
import json
from itertools import islice
import concurrent.futures


np.random.seed(42)

logger.add('logs/logs.log', level='DEBUG')


class CloudImgDatasetLoader:
    def __init__(self,
                 df: pd.DataFrame,
                 num_splits: int,
                 batch_size: int,
                 img_save_conf: json):
        r"""

        Class for dataset downloading use it when you want download a very large data,
        node that your dataset need to contain image urls(url - column),
        image id(id - column)

        Parameters
        ----------

        df : pd.Dataframe object

        num_splits : int num of dataset checkpoints, where len of one
            checkpoint is len(df) / num_splits

        batch_size : int num of images async downloading

        saved_img_quality: int quality of saved images

        img_size: Tuple(int, int) default is (256, 256) the size of saved images


        """
        self.batch_size = batch_size
        self.df_loader = self.split_df_iterator(self._dataframe_preprocessing(df), num_splits)
        self.folder = None
        self.img_folder = None
        self.dfs_folder = None
        self.save_conf = img_save_conf

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
            asyncio.run(self._load_batch(idx, subdir, batch_df))
            idx += 1

    async def _load_batch(self, idx: int, subdir: Path, batch_df: pd.DataFrame) -> None:

        self._make_dir(subdir)
        self.sub_dir = subdir
        urls, ids = batch_df['url'].to_list(), batch_df['id'].to_list()
        paths = list(map(self.generate_img_path, ids))

        def chunk(it: Iterable) -> List[Tuple[str]]:
            it = iter(it)
            return iter(lambda: tuple(islice(it, self.batch_size)), ())

        full_mime_types = []
        for batch_urls, batch_paths in tqdm(zip(chunk(urls), chunk(paths))):
            mime_types = list(map(self.file_download_task, batch_urls, batch_paths))
            mime_types = await asyncio.gather(*mime_types)
            full_mime_types += [*mime_types]

        local_csv_path = os.path.join(self.dfs_folder, f'{idx}.parquet.gzip')
        batch_df['local_path'] = paths
        batch_df['mime_type'] = full_mime_types
        batch_df.to_parquet(local_csv_path, compression='gzip')

    async def file_download_task(self, url: str, path: Path) -> str:
        data = await self._load_sample(url)
        await self._save_img(**data, local_path=path)
        return data['mime_type']

    async def _save_img(self, content: bytes, mime_type: str, local_path: Path) -> Optional[str]:
        if content and mime_type.split('/')[0] == 'image':
            try:
                image = Image.open(io.BytesIO(content)).convert("RGB")
                for save_type in self.save_conf:
                    for item in self.save_conf[save_type]:
                        size, quality = tuple(item['size']), item['quality']
                        item_path = f"{local_path}_{size[0]}_{quality}.jpg"
                        image.resize(size).save(fp=item_path, quality=quality, format='JPEG')
                return local_path
            except Exception as e:
                logger.debug(e)
                return None
        elif content:
            with open(local_path, 'wb') as f:
                f.write(content)

    def generate_img_path(self, img_id: str) -> Path:
        hasher = hashlib.sha1(str(img_id).encode('utf-8'))
        image_name = base64.urlsafe_b64encode(hasher.digest()).decode('utf-8')
        image_name = str(image_name).replace('=', '').lower()
        n = 2
        sub_dirs = [image_name[i:i + n] for i in range(0, len(image_name), n)][:3]
        res_path = self.sub_dir
        for sub_dir_img in sub_dirs:
            res_path = os.path.join(res_path, sub_dir_img)
        self._make_dir(res_path)
        res_path = os.path.join(res_path, image_name)
        return res_path

    def _save_result_df(self) -> None:

        def get_model_paths(path: str) -> Optional[str]:
            model_img_path = f'{path}_{dev_size}_{dev_q}.jpg'
            if os.path.exists(model_img_path):
                return model_img_path
            elif os.path.exists(path):
                return None
            else:
                dir_path = "/".join(path.split('/')[:-1])
                shutil.rmtree(dir_path)
                return None

        dev_conf = self.save_conf['dev'][0]
        dev_size, dev_q = dev_conf['size'][0], dev_conf['quality']

        dfs = [pd.read_parquet(os.path.join(self.dfs_folder,
                                            path)) for path in os.listdir(self.dfs_folder)]

        res_path = os.path.join(self.dfs_folder, 'full_data.parquet.gzip')
        df = pd.concat(dfs, ignore_index=True)
        paths = df.local_path.to_list()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            model_paths = list(executor.map(get_model_paths, paths))

        df['model_img_path'] = model_paths

        df.to_parquet(res_path, compression='gzip')

    @staticmethod
    def _dataframe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['id', 'url'])
        df = df.drop_duplicates(subset=['id', 'url'], keep='first', ignore_index=True)
        return df.sample(frac=1).reset_index(drop=True)

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
        data = {'content': None, 'mime_type': None}
        content = None
        if url.startswith('ipfs://'):
            url = url.replace('ipfs://', 'https://ipfs.io/ipfs/')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    content = await response.read()
        except Exception as e:
            logger.debug(e)
            logger.debug(url)
            pass
        finally:
            if not(content is None):
                data['mime_type'] = magic.from_buffer(content, mime=True)
                data['content'] = content
            return data
