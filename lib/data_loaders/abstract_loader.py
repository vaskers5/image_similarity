from abc import ABC
import os
import pandas as pd
from loguru import logger
import aiofiles
import shutil
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import hashlib
from urllib.parse import urlparse


from containers import SqlConnector
from lib.data_loaders.df_loader import Idf_loader
from lib.batch_info import CheckpointInfo


class AbstractLoader(ABC):
    def __init__(self,
                 df_iterator: Idf_loader,
                 seq_len: int,
                 batch_size: int,
                 out_dataset_path: str):
        r"""

        Class for dataset downloading use it when you want download a very large data,
        node that your dataset need to contain image urls(url - column),
        image id(id - column)

        Parameters
        ----------
        seq_len: int how many rows you want to get from df_iterator for one iteration

        df_iterator : Iterable you need to create/choose dataframe custom generator

        batch_size : int num of images async downloading

        out_dataset_path : str path for out dataset folder

        """
        self.batch_size = batch_size
        self.df_iterator = df_iterator(seq_len).load()
        self.db_store = SqlConnector.sql_connector()
        self.df_loader = Idf_loader
        self.folder = None
        self.out_dataset_path = out_dataset_path
        self.folder = os.path.abspath(out_dataset_path)
        self.src_img_folder = os.path.join(self.folder, 'img_data_src')

    def __call__(self) -> None:
        r""" Function for start downloading.

        File structure after downloading:

        -image_data: directory with folders of images

        -datasets: directory with checkpoints of dataframes saved in csv format
        """
        self._make_main_dirs()
        self._load_full_data()

    def save_checkpoint(self,
                        info: CheckpointInfo) -> None:

        info_records = info.to_records()
        cleared_src_paths = list(map(self._clear_checkpoint, info_records['path']))

        main_folder = "/".join(str(self.folder).split('/')[:-2])

        def cut_path(path: Path) -> str:
            return str(path).replace(main_folder, '')

        place_holder = [None for i in range(len(info_records['url']))]
        meta_data = []

        for idx, url in enumerate(info_records['url']):

            if len(url) >= 254:
                info_records['url'][idx] = ''
                meta_data += [url]
            else:
                meta_data += [None]
        last_id = self.db_store.get_last_bd_id('tmp_ml_token_img') + 1
        ids = list(range(last_id, last_id + len(info_records['url'])))
        save_df = pd.DataFrame({
            'id': ids,
            'filePath': list(map(cut_path, cleared_src_paths)),
            'fileData': meta_data,
            'fileType': place_holder,
            'fileRemoteUrl': info_records['url'],
            'tokenId': info_records['id'],
            'statusCode': info_records['status'],
            'fileMeta': place_holder
        })
        self.db_store.write_to_df(save_df)

    async def _load_batch(self, idx: int, batch_df: pd.DataFrame, need_saving=True) -> None:
        raise NotImplementedError

    async def _file_download_task(self,
                                  sample_id: str,
                                  url: str,
                                  src_path: Path) -> Dict[str, str]:
        task_info = {'url': url, 'path': src_path, 'id': sample_id}
        if not os.path.exists(src_path):
            content, load_status = await self._load_sample(url)
            save_status = await self._save_img(content, local_source_path=src_path)

            task_info['status'] = save_status if save_status else load_status
        else:
            task_info['status'] = 200
        return task_info

    async def _save_img(self, content: bytes, local_source_path: Path) -> None:
        if content:
            try:
                await self.save_file(local_source_path, content)
                return 200
            except Exception as e:
                return 665

    def _generate_img_path(self, img_id: str) -> Path:
        hasher = hashlib.sha1(str(img_id).encode('utf-8'))
        image_name = base64.urlsafe_b64encode(hasher.digest()).decode('utf-8')
        image_name = str(image_name).replace('=', '').lower()
        n = 2
        sub_dirs = [image_name[i:i + n] for i in range(0, len(image_name), n)][:3]
        res_src_path = self.src_img_folder
        for sub_dir_img in sub_dirs:
            res_src_path = os.path.join(res_src_path, sub_dir_img)
        final_res_path = os.path.join(res_src_path, image_name)
        if not os.path.exists(final_res_path):
            self._make_dir(res_src_path)
        return final_res_path

    def _gen_all_paths(self, ids: List[int]) -> Tuple[List[Path]]:
        src_paths = map(self._generate_img_path, ids)
        return list(src_paths)

    def _make_main_dirs(self) -> None:
        self._make_dir(self.src_img_folder)

    @staticmethod
    def _dataframe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        logger.info('Start dataframe preprocessing!')
        df = df.dropna(subset=['id', 'imageUrl'])
        df = df.drop_duplicates(subset=['id', 'imageUrl'], keep='first', ignore_index=True)
        df = df[df['imageUrl'] != ""]
        df['domen'] = list(df.imageUrl.parallel_apply(lambda url: urlparse(url).netloc))
        return df

    @staticmethod
    def _clear_checkpoint(src_path: str) -> Optional[str]:
        src_dir_path = "/".join(src_path.split('/')[:-1])
        if os.path.exists(src_path):
            return src_path
        else:
            shutil.rmtree(src_dir_path)
            return None

    @staticmethod
    async def save_file(path: str, image: memoryview) -> None:
        async with aiofiles.open(path, "wb") as file:
            await file.write(image)

    @staticmethod
    def _make_dir(dir_path: str) -> None:
        if not(os.path.exists(dir_path)):
            os.makedirs(dir_path)
