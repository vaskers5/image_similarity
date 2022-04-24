import pickle
import pandas as pd
import torchvision.models as models
import json

from lib.lsh_table import ForestInferer
from lib.models import FeatureExtractor


if __name__ == "__main__":
    df = pd.read_csv('data/test_data.csv')
    model = models.vgg19(pretrained=True)
    feature_model = FeatureExtractor(model, layers=['classifier.5'])

    with open('checkpoints/tree_db.pt', 'rb') as f:
        tree = pickle.loads(f.read())
    tree.index()

    forest = ForestInferer(df=df,
                           model=feature_model,
                           device='cuda',
                           batch_size=100,
                           tree=tree,
                           num_permutations=50)
    forest_result = forest.start_infer(10)
    with open('data/duplicates_finder.json', 'w', encoding='utf-8') as f:
        json.dump(forest_result, f, ensure_ascii=False, indent=4)


