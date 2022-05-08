from datasketch import MinHashLSHForest, MinHash
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm


def get_minhash(name: str) -> MinHash:
    m_hash = MinHash(num_perm=128)
    for d in name.split():
        m_hash.update(d.encode('utf8'))
    return m_hash


if __name__ == "__main__":
    df = pd.read_csv('data/checknft_collections.csv')
    forest = MinHashLSHForest(num_perm=128)
    full_names_ds = df['name'].dropna().to_list()
    
    with Pool() as pool:
        hash_values = list(tqdm(pool.map(get_minhash, full_names_ds),
                                total=len(full_names_ds)))
    
    for name, hash_value in tqdm(zip(full_names_ds, hash_values)):
        forest.add(name, hash_value)
        
    with open('checkpoints/tree_db_words_data.pt', 'wb') as f:
        f.write(pickle.dumps(tree))
