""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import segmentation_models_pytorch as smp

def createUnet(outputchannels=1):
    '''
    Unet Decoder with custom head and ResNet18 Encoder

    Args:
        outputchannels (int, optional):
            num output channels in the dataset masks
    Returns:
        pytorch model - Unet decoder with ResNet18 encoder
    '''
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=1,
        classes=outputchannels
    )
    
    model.train()
    
    return model 


def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
