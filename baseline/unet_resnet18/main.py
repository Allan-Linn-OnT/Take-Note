from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
from torchvision import transforms

import datahandler
from model import createUnet, createFPN
from trainer import train_model


@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--arch",
              default='Unet',
              type=str,
              help="Specify the Arch of the model")

def main(data_directory, exp_directory, epochs, batch_size, arch):
    
    model = createUnet() if arch == 'Unet' else createFPN()
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, }#'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_sep_folder(
        data_directory,
        image_folder='input',
        mask_folder='target',
        batch_size=batch_size,
        data_transforms=[
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize(size=(512,512)),
            ]
        )
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'best_weights.pt')


if __name__ == "__main__":
    main()
