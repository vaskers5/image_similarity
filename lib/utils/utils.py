import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from typing import List
from tqdm import tqdm


def train_step(encoder: nn.Module,
               decoder: nn.Module,
               train_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: nn.Module,
               device: str) -> float:
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    decoder(: A convolutional Decoder. E.g. torch_model ConvDecoder16
    train_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    optimizer: PyTorch optimizer.
    device: "cuda" or "cpu"
    Returns: Train Loss
    """

    encoder.train()
    decoder.train()

    for (train_img, target_img) in tqdm(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        optimizer.zero_grad()
        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)
        loss = loss_fn(dec_output, target_img)
        loss.backward()
        optimizer.step()

    return loss.item()


def val_step(encoder: nn.Module,
             decoder: nn.Module,
             val_loader: DataLoader,
             loss_fn: nn.Module,
             device) -> float:
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    decoder: A convolutional Decoder. E.g. torch_model ConvDecoder
    val_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    device: "cuda" or "cpu"
    Returns: Validation Loss
    """
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for (train_img, target_img) in tqdm(val_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)
            loss = loss_fn(dec_output, target_img)
    return loss.item()



def create_embedding(encoder: nn.Module, full_loader: DataLoader, embedding_dim: np.ndarray, device: str):
    """
    Creates embedding using encoder from dataloader.
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimesntions.
    device: "cuda" or "cpu"
    Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
    """

    encoder.eval()
    # Just a place holder for our 0th image embedding.
    embedding = torch.randn(embedding_dim)
    
    with torch.no_grad():
        for (train_img, target_img) in tqdm(full_loader):
            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)

    return embedding


def compute_similar_images(image: Image,
                           encoder: nn.Module,
                           num_images: int,
                           embedding: np.ndarray,
                           device: str='cpu') -> List[List[int]]:
    """
    Given an image and number of similar images to search.
    Returns the num_images closest neares images.
    Args:
    image: Image whose similar images are to be found.
    ecnoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """
    
    image_tensor = T.ToTensor()(image)
    image_tensor = T.Resize([512, 512])(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    return indices_list
