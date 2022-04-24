import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm_notebook
import torch
import torchvision.transforms as T

from lib.dataset import FolderDataset
from lib.models import ConvEncoder, ConvDecoder
from lib.utils import train_step, val_step, create_embedding

pandarallel.initialize(progress_bar=True)


def check_is_img(path: str) -> bool:
    try:
        Image.open(path)
        return True
    except:
        return False


if __name__=='__main__':
    dataset_path = '/mnt/0806a469-d019-4d6a-be45-7cff5d66eb22/datasets/image_similarity_set/full_data.csv'
    df = pd.read_csv(dataset_path).dropna(subset=['id', 'url'])
    checks = df.local_path.parallel_apply(check_is_img)
    df = df[checks]
    local_paths = df['local_path'].to_list()[:800000]

    transforms = T.Compose([T.ToTensor(), T.Resize([512, 512])])  # Normalize the pixels and convert to tensor.

    full_dataset = FolderDataset(local_paths, transforms)  # Create folder dataset.

    batch_size = 65
    val_size = 50000
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [800000 - val_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size)

    loss_fn = nn.MSELoss()

    encoder = ConvEncoder()
    decoder = ConvDecoder()

    device = "cuda"

    encoder.to(device)
    decoder.to(device)

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(autoencoder_params, lr=1e-3)

    EPOCHS = 30

    max_loss = float('inf')

    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)

        print(f"Epochs = {epoch}, Training Loss : {train_loss}")

        val_loss = val_step(encoder, decoder, val_loader, loss_fn, device=device)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

        if val_loss < max_loss:
            max_loss = val_loss
            print("Validation Loss decreased, saving new best model")
            torch.save(encoder.state_dict(), f"checkpoints/encoder_model_{epoch}.pt")
            torch.save(decoder.state_dict(), f"checkpoints/decoder_model_{epoch}.pt")
