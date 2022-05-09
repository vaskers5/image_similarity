from lib.data_processing import CloudImgDatasetLoader
import pandas as pd
import json

if __name__ == "__main__":
    with open('configs/img_save_conf.json') as f:
        conf = json.load(f)

    # folder_path = '/mnt/0806a469-d019-4d6a-be45-7cff5d66eb22/datasets/60m_tokens_set_1'
    # dataset_path = f'{folder_path}/30m_data.parquet.gzip'
    # df = pd.read_parquet(f'{folder_path}/30m_data.parquet.gzip')

    folder_path = 'data/test_loading_folder'
    dataset_path = 'data/test_infer.csv'
    df = pd.read_csv(dataset_path)
    del df['local_path']

    loader = CloudImgDatasetLoader(df, **conf)
    loader(folder_path)
