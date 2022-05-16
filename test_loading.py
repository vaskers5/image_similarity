from lib.data_loaders import CloudImgDatasetLoader
import pandas as pd
import json
import os

from lib.data_loaders.df_loader import DbDfLoader

if __name__ == "__main__":
    with open('configs/img_save_conf.json') as f:
        conf = json.load(f)
    loader = CloudImgDatasetLoader(df_iterator=DbDfLoader, **conf)
    loader()
