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
import random
from loguru import logger
import json
from itertools import islice
import concurrent.futures
import aiofiles
from aiohttp_retry import RetryClient, FibonacciRetry




np.random.seed(42)

logger.add('logs/logs.log', level='DEBUG')


def gen_proxy(proxy_id: int) -> str:
    super_proxy_url = ('http://%s-session-%s:%s@zproxy.lum-superproxy.io:%d' %
                       (os.getenv('PROXY_USER'),
                        proxy_id,
                        os.getenv('PROXY_PASSWORD'),
                        int(os.getenv("PROXY_PORT"))))
    return super_proxy_url


ALL_PROXY = [gen_proxy(i) for i in range(100)]
ALL_PROXY.append('self_url')


class CloudImgDatasetLoader:
    def __init__(self,
                 df: pd.DataFrame,
                 num_splits: int,
                 batch_size: int,
                 out_dataset_path: str,
                 img_save_conf: json,
                 last_checkpoint_num: int = None):
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

        img_save_conf:
            saved_img_quality: int quality of saved images

            img_size: Tuple(int, int) default is (256, 256) the size of saved images

        out_dataset_path : str path for out dataset folder

        """
        self.batch_size = batch_size
        self.df_loader = self.split_df_iterator(self._dataframe_preprocessing(df), num_splits)
        self.folder = None
        self.img_folder = None
        self.dfs_folder = None
        self.save_conf = img_save_conf
        self.out_dataset_path = out_dataset_path
        self.folder = os.path.abspath(out_dataset_path)
        self.img_folder = os.path.join(self.folder, 'img_data')
        self.src_img_folder = os.path.join(self.folder, 'img_data_src')
        self.dfs_folder = os.path.join(self.folder, 'datasets')
        self.last_checkpoint_num = last_checkpoint_num

    def __call__(self) -> None:
        r""" Function for start downloading.

        File structure after downloading:

        -image_data: directory with folders of images

        -datasets: directory with checkpoints of dataframes saved in csv format
        """
        self._make_main_dirs()
        self._load_full_data(self.last_checkpoint_num)
        self._save_result_df()

    def _make_main_dirs(self) -> None:
        self._make_dir(self.img_folder)
        self._make_dir(self.dfs_folder)
        self._make_dir(self.dfs_folder)

    def _load_full_data(self, checkpoint_num: Optional[int]) -> None:
        idx = -1
        logger.info('Start downloading loop')
        for batch_df in tqdm(self.df_loader):
            idx += 1
            if checkpoint_num and idx < checkpoint_num:
                logger.info(f'Skip {idx} because of previous downloading')
                continue
            elif checkpoint_num == idx:
                logger.info(f'Downloading left images on {checkpoint_num}')
                self._load_only_left(idx, batch_df)
            else:
                logger.info(f'Checkpoint {idx} on downloading')
                asyncio.run(self._load_batch(idx, batch_df))
                logger.info(f'Checkpoint {idx} was successfully downloaded')

    async def _load_batch(self, idx: int, batch_df: pd.DataFrame, need_saving=True) -> None:
        urls, ids = batch_df['url'].to_list(), batch_df['id'].to_list()
        image_paths, src_paths = self._gen_all_paths(ids)

        def chunk(it: Iterable) -> List[Tuple[str]]:
            it = iter(it)
            return iter(lambda: tuple(islice(it, self.batch_size)), ())

        for batch_urls, batch_image_paths, batch_src_paths in tqdm(zip(chunk(urls),
                                                                       chunk(image_paths),
                                                                       chunk(src_paths))):
            await asyncio.wait(map(self._file_download_task,
                                   batch_urls,
                                   batch_image_paths,
                                   batch_src_paths))

        if need_saving:
            self.save_checkpoint(idx, batch_df, image_paths, src_paths)

    def save_checkpoint(self,
                        idx: int,
                        batch_df: pd.DataFrame,
                        image_paths: List[Path],
                        src_paths: List[Path]):
        local_csv_path = os.path.join(self.dfs_folder, f'{idx}.parquet.gzip')
        batch_df['local_path'] = image_paths
        batch_df['src_path'] = src_paths
        batch_df['checkpoint'] = [idx for i in range(len(image_paths))]
        batch_df.to_parquet(local_csv_path, compression='gzip')

    async def _file_download_task(self, url: str, path: Path, src_path: Path) -> None:
        content = await self._load_sample(url)
        await self._save_img(content, local_path=path, local_source_path=src_path)

    async def _save_img(self, content: bytes, local_path: Path, local_source_path: Path) -> None:
        if content:
            try:
                await self.save_file(local_source_path, content)
                #self.save_file(local_source_path, content)
            # with open(local_source_path, 'wb') as f:
            #     f.write(content)
            # try:
            #     image = Image.open(io.BytesIO(content)).convert("RGB")
            #     for save_type in self.save_conf:
            #         for item in self.save_conf[save_type]:
            #             size, quality = tuple(item['size']), item['quality']
            #             item_path = f"{local_path}_{size[0]}_{quality}.jpg"
            #             image = image.resize(size)
            #             image.save(item_path, quality=quality, format='JPEG')
            except Exception as e:
                logger.debug(e)
                return None

    def _generate_img_path(self, img_id: str) -> Tuple[Path, Path]:
        hasher = hashlib.sha1(str(img_id).encode('utf-8'))
        image_name = base64.urlsafe_b64encode(hasher.digest()).decode('utf-8')
        image_name = str(image_name).replace('=', '').lower()
        n = 2
        sub_dirs = [image_name[i:i + n] for i in range(0, len(image_name), n)][:3]
        res_path, res_src_path = self.img_folder, self.src_img_folder
        for sub_dir_img in sub_dirs:
            res_path = os.path.join(res_path, sub_dir_img)
            res_src_path = os.path.join(res_src_path, sub_dir_img)
        #self._make_dir(res_path)
        self._make_dir(res_src_path)
        res_path = os.path.join(res_path, image_name)
        res_src_path = os.path.join(res_src_path, image_name)
        return res_path, res_src_path

    def _load_only_left(self, idx, batch_df: pd.DataFrame) -> None:
        ids = batch_df['id'].to_list()
        image_paths, src_paths = self._gen_all_paths(ids)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(executor.map(self._clear_checkpoint, image_paths, src_paths))

        loaded_image_paths = [item[0] for item in data]
        part_df = batch_df[pd.isna(loaded_image_paths)]
        asyncio.run(self._load_batch(idx, part_df, need_saving=False))
        self.save_checkpoint(idx, batch_df, image_paths, src_paths)

    def _save_result_df(self) -> None:

        dfs = [pd.read_parquet(os.path.join(self.dfs_folder,
                                            path)) for path in os.listdir(self.dfs_folder)]

        res_path = os.path.join(self.dfs_folder, 'full_data.parquet.gzip')
        df = pd.concat(dfs, ignore_index=True)
        paths, src_paths = df.local_path.to_list(), df.src_path.to_list()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = executor.map(self._clear_checkpoint, paths, src_paths)

        model_img_paths, src_paths = zip(*data)
        df['model_img_path'] = list(model_img_paths)
        df['src_path'] = list(src_paths)
        df.to_parquet(res_path, compression='gzip')

    def _clear_checkpoint(self, path: str, src_path: str) -> Optional[str]:
        dev_conf = self.save_conf['dev'][0]
        dev_size, dev_q = dev_conf['size'][0], dev_conf['quality']
        model_img_path = f'{path}_{dev_size}_{dev_q}.jpg'
        image_dir_path = "/".join(path.split('/')[:-1])
        src_dir_path = "/".join(src_path.split('/')[:-1])
        # if os.path.exists(model_img_path):
        #     return model_img_path, src_path
        if os.path.exists(src_path):
            #shutil.rmtree(image_dir_path)
            return path, src_path
        else:
            shutil.rmtree(src_dir_path)
            #shutil.rmtree(image_dir_path)
            return None, None

    def _gen_all_paths(self, ids: List[int]) -> Tuple[List[Path]]:
        data = map(self._generate_img_path, ids)
        image_paths, src_paths = zip(*data)
        return list(image_paths), list(src_paths)

    @staticmethod
    def _dataframe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['id', 'url'])
        df = df.drop_duplicates(subset=['id', 'url'], keep='first', ignore_index=True)
        df = df[df['imageUrl'] != ""]
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    @staticmethod
    def split_df_iterator(df: pd.DataFrame, chunks: int) -> Iterable[pd.DataFrame]:
        for chunk in np.array_split(df, chunks):
            yield chunk

    @staticmethod
    def _make_dir(dir_path: str) -> None:
        if not(os.path.exists(dir_path)):
            os.makedirs(dir_path)

    @staticmethod
    async def _load_sample(url: str) -> Optional[bytes]:
        content = None
        headers = None
        proxy_url = random.choice(ALL_PROXY)
        if proxy_url == 'self_url':
            proxy_url = None
        if url.startswith('ipfs://'):
            url = url.replace('ipfs://', 'https://ipfs.io/ipfs/')
        elif 'pinata' in url:
            headers = {
                'pinata_api_key': os.getenv('PINATA_API_KEY'),
                'pinata_secret_api_key': os.getenv('PINATA_SECRET')
            }
            proxy_url = None
        try:
            async with RetryClient(retry_options=FibonacciRetry(attempts=5)) as session:
                async with session.get(url, proxy=proxy_url, ssl=False, headers=headers) as response:
                    content = await response.read()
        except Exception as e:
            logger.debug(e)
            logger.debug(url)
            pass
        finally:
            return content

    @staticmethod
    async def save_file(path: str, image: memoryview) -> None:
        async with aiofiles.open(path, "wb") as file:
            await file.write(image)

    # @staticmethod
    # def save_file(path: str, image: bytes) -> None:
    #     with open(path, "wb") as file:
    #         file.write(image)
