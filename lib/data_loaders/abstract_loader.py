from abc import ABC
import os
import pandas as pd
import json
import aiofiles
import shutil
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import hashlib


from containers import SqlConnector
from lib.data_loaders.df_loader import Idf_loader


class AbstractLoader(ABC):
    def __init__(self,
                 df_iterator: Idf_loader,
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

        df_iterator : Iterable you need to create/choose dataframe custom generator

        batch_size : int num of images async downloading

        img_save_conf:
            saved_img_quality: int quality of saved images

            img_size: Tuple(int, int) default is (256, 256) the size of saved images

        out_dataset_path : str path for out dataset folder

        """
        self.batch_size = batch_size
        self.df_iterator = df_iterator
        self.db_store = SqlConnector.sql_connector()
        self.df_loader = Idf_loader
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

    def save_checkpoint(self,
                        batch_df: pd.DataFrame,
                        src_paths: List[Path],
                        status_codes: List[int]):

        with ThreadPoolExecutor() as executor:
            src_paths = executor.map(self._clear_checkpoint, src_paths)

        main_folder = "/".join(str(self.folder).split('/')[:-2])

        def cut_path(path: Path) -> str:
            return str(path).replace(main_folder, '')

        place_holder = [None for i in range(len(batch_df))]
        save_df = pd.DataFrame({
            'filePath': list(map(cut_path, src_paths)),
            'fileData': place_holder,
            'fileType': place_holder,
            'fileRemoteUrl': batch_df['url'].to_list(),
            'tokenId': batch_df['id'].astype(int).to_list(),
            'statusCode': status_codes
        })
        self.db_store.write_to_df(save_df)

    async def _load_batch(self, idx: int, batch_df: pd.DataFrame, need_saving=True) -> None:
        raise NotImplementedError

    async def _file_download_task(self, url: str, path: Path, src_path: Path) -> None:
        if not os.path.exists(src_path):
            content = await self._load_sample(url)
            await self._save_img(content, local_path=path, local_source_path=src_path)

    async def _save_img(self, content: bytes, local_source_path: Path) -> None:
        if content:
            await self.save_file(local_source_path, content)

    def _generate_img_path(self, img_id: str) -> Tuple[Path, Path]:
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
        data = map(self._generate_img_path, ids)
        image_paths, src_paths = zip(*data)
        return list(image_paths), list(src_paths)

    def _make_main_dirs(self) -> None:
        self._make_dir(self.img_folder)
        self._make_dir(self.dfs_folder)
        self._make_dir(self.dfs_folder)

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
