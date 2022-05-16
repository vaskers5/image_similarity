import pandas as pd

from lib.data_loaders import AbstractLoader


class IpfsLoader(AbstractLoader):
    async def _load_batch(self, idx: int, batch_df: pd.DataFrame, need_saving=True) -> None:
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
