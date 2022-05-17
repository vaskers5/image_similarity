from typing import Iterable

from .Idf_loader import IDfLoader
from containers import SqlConnector


class DbDfLoader(IDfLoader):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.db_con = SqlConnector.sql_connector()
        self.last_id = self.db_con.get_last_bd_id('ml_token_img')

    def load(self) -> Iterable:
        for i in range(1, self.last_id, self.seq_len):
            yield self.db_con.read_rows(start=i, seq_len=self.seq_len)
