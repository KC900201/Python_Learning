import torch
import os
import pandas as pd
from PIL import Image

class VOC(torch.utils.data.Dataset):
    def __init__(self):