from typing import Iterable

from .Idf_loader import IDfLoader
from containers import SqlConnector


class DbDfLoader(IDfLoader):
    def load(self, batch_size: int = 1000) -> Iterable:
        db_con = SqlConnector.sql_connector()
        for i in range(1, db_con.get_last_bd_id(), batch_size):
            yield db_con.read_rows(offset=i, n_rows=batch_size)
