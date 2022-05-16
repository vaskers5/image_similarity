from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
import psycopg2


sql = """
with token_data as (select t.id, "name", "description"
    from token as t
    order by id
    limit 1000),
    t_d as (select id, "imageUrl", "collectionId"
    from ml_token_img
    where "imageUrl" != ''
    order by id
    limit 1000)

select td.id, "name", "description", "imageUrl", "collectionId"
from token_data as td
inner join t_d as td1
on td.id=td1.id

"""


con_args = {'drivername': 'postgresql',
            'host': '94.130.201.172',
            'username': 'mlremoteuser',
            'password': 'LJyRmZEfBae',
            'database': 'checknft',
            'port': 5432,}


if __name__ == "__main__":
    engine = create_engine(URL(**con_args))
    dat = pd.read_sql_query(sql, engine)
    path = '/mnt/0806a469-d019-4d6a-be45-7cff5d66eb22/datasets/60m_tokens_set_1/30m_data.parquet.gzip'
    dat.to_parquet(path, index=False, compression='gzip')