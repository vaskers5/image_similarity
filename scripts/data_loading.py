import pandas as pd
from typing import Dict
import argparse
import os

from lib.data_processing import CloudImgDatasetLoader


def load_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser(description='Type path for dataset and out data')
    parser.add_argument('-dataset', type=str, help='Input csv path')
    parser.add_argument('-out_dir', type=str, help='Output image data folder')
    return parser.parse_args()


# call example: python data_loading.py -dataset data/test.csv -out_dir data/full_data

if __name__=="__main__":

    args = load_args()
    print(args)

    df = pd.read_csv(args.dataset)
    output_dir = args.out_dir

    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    loader = CloudImgDatasetLoader(df)
    result_df = loader(output_dir)

    result_df.to_csv(f'{output_dir}/full_data.csv', index=False)
