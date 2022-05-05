import torch
import pandas as pd
import pickle
import torchvision.models as models

from lib.models import FeatureExtractor
from lib.dataset import FolderDataset
from lib.lsh_table import HashBuilderFromModel
from lib.utils.transforms_list import TRANSFORMS_LIST

if __name__ == "__main__":
    #df = pd.read_csv('data/test_infer.csv')
    df = pd.read_parquet('data/new_full_data_90.parquet.gzip')
    all_imgs = df.local_paths.to_list()
    ids = df.id.to_list()
    transforms = TRANSFORMS_LIST

    full_dataset = FolderDataset(all_imgs, ids, transforms)
    batch_size = 128
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size)

    torch.cuda.empty_cache()

    model = models.resnet152(pretrained=True)
    device = 'cuda'
    feature_model = FeatureExtractor(model, layers=['avgpool'])
    feature_model.eval()
    feature_model.to(device)

    table = HashBuilderFromModel(feature_model, 'cuda', full_dataset, 256, 12, 0.90)
    tree, lsh_table, hash_gen = table.build_forest(full_loader)

    with open('checkpoints/tree5_db_new_data_resnet152.pt', 'wb') as f:
        f.write(pickle.dumps(tree))

    with open('checkpoints/lsh5_db_new_data_resnet152.pt', 'wb') as f:
        f.write(pickle.dumps(lsh_table))

    with open('checkpoints/hash5_gen_new_data_resnet152.pt', 'wb') as f:
        f.write(pickle.dumps(hash_gen))
