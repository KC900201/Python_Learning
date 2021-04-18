import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of output channels in a backbone. For mobilenet_v2, it's 1280
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 1.5, 2.0),))

# Let's define what are the feature maps that we will use to perform the region of interest cropping
# , as well as the size of the crop after rescaling.
# if your backbone returs a Tensor, featmap_games is expected to be [0]. More_generally, the backbone
# should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# put the pieces together inside a FasterRCNN Model
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


