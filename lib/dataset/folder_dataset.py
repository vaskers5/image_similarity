from torch.utils.data.dataset import Dataset
from PIL import Image
from typing import List


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, all_imgs: List[str], uniq_indexes: List[str], transform=None):
        self.transform = transform
        self.all_imgs = all_imgs
        self.uniq_indexes = uniq_indexes

    def __len__(self):
        return len(self.all_imgs)

    def get_id(self, idx: int) -> str:
        return self.uniq_indexes[idx]

    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image
