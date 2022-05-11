from sqlalchemy import create_engine
import os
import pandas as pd


class DBStore:
    def __init__(self):
        self.engine = create_engine(os.getenv('DB_URI'))

    def read_rows(self, n_rows: int, offset: int):
        ids = list(range(offset, offset + n_rows))
        query = f"""
        select
            id, "imageUrl"
        from ml_token_img
        where id IN ({",".join(ids)})
        """
        return pd.read_sql_query(query, self.engine)

    def write_to_df(self):
        pass
