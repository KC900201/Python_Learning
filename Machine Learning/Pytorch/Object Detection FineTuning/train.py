import torch.utils.data
import torchvision

import utils
from dataset import PennFudanDataset
from utils import get_transform


def train():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn
    )
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions

# Main function for training
def main():
    num_classes = 2
    # use our dataset and defined transformation
    dataset = PennFudanDataset() # Continue


if __name__ == '__main__':
    train()
