from sqlalchemy import create_engine
import pandas as pd


class DBStore:
    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri)
        self.engine.autocommit = True

    def read_rows(self, start: int, seq_len: int) -> pd.DataFrame:
        query = f"""
        select
            id, "imageUrl"
        from ml_token_img
        where id between {start} and {start + seq_len}
        """
        return pd.read_sql_query(query, self.engine)

    def get_last_bd_id(self):
        query = f"""
        select id
        from ml_token_img
        order by id desc
        limit 1
        """
        return int(pd.read_sql_query(query, self.engine).id.iloc[0])

    def write_to_df(self, df: pd.DataFrame):
        df.to_sql("tmp_ml_token_img", con=self.engine, index=False, if_exists="append")
