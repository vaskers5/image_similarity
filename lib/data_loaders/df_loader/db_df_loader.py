from typing import Iterable

from .Idf_loader import IDfLoader
from containers import SqlConnector


class DbDfLoader(IDfLoader):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def load(self) -> Iterable:
        db_con = SqlConnector.sql_connector()
        for i in range(1, db_con.get_last_bd_id(), self.seq_len):
            yield db_con.read_rows(start=i, seq_len=self.seq_len)
