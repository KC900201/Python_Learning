import torch
import transforms as T

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def collate_fn(): # Continue writing src code
    pass
