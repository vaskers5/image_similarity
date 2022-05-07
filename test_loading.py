from lib.data_processing import CloudImgDatasetLoader
import pandas as pd

if __name__ == "__main__":
    folder_path = '/mnt/0806a469-d019-4d6a-be45-7cff5d66eb22/datasets/60m_tokens_set_1'

    df = pd.read_parquet(f'{folder_path}/30m_data.parquet.gzip')
    loader = CloudImgDatasetLoader(df, 600)
    loader(folder_path)
