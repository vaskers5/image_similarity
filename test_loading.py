from lib.data_processing import CloudImgDatasetLoader
import pandas as pd
import json
import os


if __name__ == "__main__":
    with open('configs/ipfs_save_conf.json') as f:
        conf = json.load(f)
    #/mnt/97a55efc-92bc-452a-98d5-d4e3e9dad536/datasets/60m_tokens_ds/full_data_clear.parquet.gzip
    folder_path = '/mnt/97a55efc-92bc-452a-98d5-d4e3e9dad536/datasets/60m_tokens_ds'
    dataset_path = f'{folder_path}/full_data_1.parquet.gzip'
    df = pd.read_parquet(f'{folder_path}/full_data_clear.parquet.gzip')

    # folder_path = 'data/test_loading_folder'
    # dataset_path = 'data/test_infer.csv'
    # df = pd.read_csv(dataset_path)
    # del df['local_path']
    if 'url' not in df.columns:
        df['url'] = df['imageUrl']
    loader = CloudImgDatasetLoader(df, **conf)
    loader.load_ipfs_samples()
