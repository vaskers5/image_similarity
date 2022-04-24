import torchvision.transforms as T


TRANSFORMS_LIST = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
