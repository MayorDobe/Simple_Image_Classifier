import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tool_box.tool_box import ImageDataSetTemplate, ModelTemplate, Trainer

# TODO: add config file.json
# TODO: argparse for command line args

config = {
    "model": "efficientnet_b0",
    "optimizer": optim.Adam,
    "criterion": nn.CrossEntropyLoss(),
    "batch_size": 32,
    "num_epochs": 6,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main() -> None:
    # Data preparation
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # Create train dataset
    train_dataset = ImageDataSetTemplate(
        "../card_classifier/dataset/train", transform=transform
    )
    # Create training loader
    training_loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True
    )

    # Create validation dataset
    valid_dataset = ImageDataSetTemplate(
        "../card_classifier/dataset/valid", transform=transform
    )
    # Create validation loader
    validation_loader = DataLoader(
        valid_dataset, batch_size=int(config["batch_size"]), shuffle=False
    )

    # Create model
    model = ModelTemplate(
        # Load base model
        model=timm.create_model(
            config["model"], pretrained=True, num_classes=train_dataset.num_classes
        ),
        # Create classifier layer num outputs to num_classes, ModelTemplate strips original classification layer
        classifier=nn.Linear(1280, train_dataset.num_classes),
    )

    # Training configuration
    trainer = Trainer(
        model=model,
        optimizer=config["optimizer"](model.parameters(), lr=config["learning_rate"]),
        criterion=config["criterion"],
        device=config["device"],
    )

    # Train model
    trainer.train_loop(training_loader, validation_loader, config["num_epochs"])

    # Save model
    model_path = "model.pt"
    torch.save(model.state_dict(), model_path)

    # Plot loss
    trainer.plot_loss()


if __name__ == "__main__":
    main()
