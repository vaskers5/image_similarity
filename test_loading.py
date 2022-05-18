from lib.data_loaders import CloudImgDatasetLoader

from lib.data_loaders.df_loader import DbDfLoader

if __name__ == "__main__":
    loader = CloudImgDatasetLoader(seq_len=100,
                                   batch_size=10,
                                   out_dataset_path="/media/hdd/1_st_dataset",
                                   df_iterator=DbDfLoader)
    loader()
