import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import asyncio
import random
from loguru import logger
from aiohttp_retry import RetryClient, FibonacciRetry
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pandarallel import pandarallel
from typing import Optional, Iterable
import re

from lib.data_loaders.abstract_loader import AbstractLoader
from lib.batch_info import CheckpointInfo

HOST_IPFS_URL = "http://212.98.190.240:8080/ipfs"

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

np.random.seed(42)

logger.add('logs/logs.log', level='DEBUG')


class CloudImgDatasetLoader(AbstractLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_full_data(self) -> None:
        logger.info('Start downloading loop')
        for idx, batch_df in enumerate(self.df_iterator):
            batch_df = self._dataframe_preprocessing(batch_df)
            logger.info(f'Checkpoint {idx} on downloading')
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._load_batch(batch_df))
            logger.info(f'Checkpoint {idx} was successfully downloaded')

    async def _load_batch(self, batch_df: pd.DataFrame) -> CheckpointInfo:
        for idx, stratified_batch in enumerate(self._get_stratified_batches(batch_df,
                                                                            self.batch_size)):
            info = CheckpointInfo()
            logger.info(f'Start downloading batch {idx}')
            urls, ids = stratified_batch['imageUrl'].to_list(), stratified_batch['id'].to_list()
            src_paths = self._gen_all_paths(ids)
            results = await asyncio.wait(map(self._file_download_task, ids, urls, src_paths),
                                         return_when=asyncio.ALL_COMPLETED)
            batch_info = [result.result() for result in results[0]]
            info.add_info(batch_info)
            logger.info(f'Start writing {len(batch_info)} rows data to DB')
            self.save_checkpoint(info)
            logger.info(f'Checkpoint {idx} was downloaded!')
        return info

    @staticmethod
    def _get_stratified_batches(df: pd.DataFrame, batch_size: int) -> Iterable:
        def get_batch(state_df: pd.DataFrame, micro_batch_size: int) -> pd.DataFrame:
            x_train, x_test, y_train, y_test = train_test_split(state_df,
                                                                state_df['domen'].to_list(),
                                                                test_size=micro_batch_size,
                                                                random_state=42)
            return x_test

        batched_df = deepcopy(df)
        batches = []
        iter_num = int(len(batched_df) / batch_size)
        for i in range(iter_num):
            if len(batched_df) > batch_size:
                batch = get_batch(batched_df, batch_size)
            else:
                batch = deepcopy(batched_df)
            batches += [batch]
            batched_df = batched_df[-batched_df['id'].isin(batch['id'].to_list())]
            yield batch
        if len(batched_df) > 0:
            yield batched_df

    async def _load_sample(self, url: str) -> Optional[bytes]:
        content = None
        if len(url) > 255:
            return content, 400
        headers = None
        proxy_url = self.proxy.refresh_url()
        old_url = deepcopy(url)
        if proxy_url == 'self_url':
            proxy_url = None
        if 'ipfs' in url:
            ipfs_id = self.parse_ipfs_url(url)
            url = f"{HOST_IPFS_URL}/{ipfs_id}"
            proxy_url = None
        if 'pinata.cloud' in url:
            headers = {
                'pinata_api_key': os.getenv('PINATA_API_KEY'),
                'pinata_secret_api_key': os.getenv('PINATA_SECRET')
            }
            url = url.replace('pinata.cloud', 'pixelplex.mypinata.cloud')
            proxy_url = None
        try:
            options = FibonacciRetry(attempts=3, max_timeout=30)
            async with RetryClient(retry_options=options) as session:
                async with session.get(url, proxy=proxy_url, ssl=False, headers=headers) as response:
                    content = await response.read()
                    status = response.status
        except Exception as e:
            # logger.warning(e)
            # logger.warning(old_url)
            return content, 400
        return content, status

    @staticmethod
    def parse_ipfs_url(url: str) -> str:

        def _parse_ipfs_url(mime_url: str) -> str:
            ipfs_reg = r'(ipfs://)|(ipfs/)|(ipfs.io/)|(ipfs.io/ipfs/)'
            parsed_url = re.search(ipfs_reg, mime_url)
            if not parsed_url:
                return mime_url
            return mime_url[parsed_url.span()[-1]:]

        for i in range(3):
            if 'ipfs' in url:
                url = _parse_ipfs_url(url)
            else:
                return url
