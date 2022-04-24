import torch
import torchvision.transforms as T
import pandas as pd
import pickle

from lib.models import FeatureExtractor
from lib.dataset import FolderDataset
from lib.lsh_table import HashBuilderFromModel
import torchvision.models as models


if __name__ == "__main__":
    df = pd.read_csv('data/full_clear_data.csv')
    all_imgs = df.local_path.to_list()
    ids = df.id.to_list()
    transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    full_dataset = FolderDataset(all_imgs, ids, transforms)
    batch_size = 100
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size)

    model = models.vgg19(pretrained=True)

    feature_model = FeatureExtractor(model, layers=['classifier.5'])

    table = HashBuilderFromModel(feature_model, 'cuda', full_dataset, 50, 10, 0.75)
    tree, min_hash = table.build_forest(full_loader)
    with open('checkpoints/tree_db.pt', 'wb') as f:
        f.write(pickle.dumps(tree))
    with open('checkpoints/lsh_db.pt', 'wb') as f:
        f.write(pickle.dumps(min_hash))

