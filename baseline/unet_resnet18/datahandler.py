from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset


def get_dataloader_sep_folder(data_dir: str,
                              image_folder: str = 'input',
                              mask_folder: str = 'target',
                              batch_size: int = 4,
                              data_transforms = [transforms.ToTensor()]
                              ):
    """ Create train and valid dataloaders from two
        separate train and valid folders.
        The directory structure should be as follows.
        data_dir
        --train - N.B Also includes validation data
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --valid
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM

    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        train and valid dataloaders.
    """
    composed_transforms = transforms.Compose(data_transforms)

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=composed_transforms,
                               image_folder=image_folder,
                               mask_folder=mask_folder)
        for x in ['train_image', 'valid_image']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=12)
        for x in ['train_image', 'valid_image']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'Images',
                                 mask_folder: str = 'Masks',
                                 fraction: float = 0.2,
                                 batch_size: int = 4,
                                 data_transforms = [transforms.ToTensor()]):
    """Create train and valid dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of valid set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        train and valid dataloaders.
    """
    composed_transforms = transforms.Compose(data_transforms)

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=composed_transforms)
        for x in ['train', 'valid']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=1)
        for x in ['train', 'valid']
    }
    return dataloaders
