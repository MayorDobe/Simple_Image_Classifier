import timm
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from tool_box.tool_box import Evaluater, ImageDataSetTemplate, ModelTemplate

# TODO: add config file.json
# TODO: argparse for command line args

device = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    eval_dataset = ImageDataSetTemplate(
        "../card_classifier/dataset/test", transform=transform
    )

    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    model = ModelTemplate(
        model=timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=eval_dataset.num_classes
        ),
        classifier=nn.Linear(1280, eval_dataset.num_classes),
    )

    # Create validation dataset
    valid_dataset = ImageDataSetTemplate(
        "../card_classifier/dataset/valid", transform=transform
    )
    # Create validation loader
    validation_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False
    )
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.to(device)

    evaluater = Evaluater(
        model=model,
        dataset=eval_dataset,
        device=device,
    )
    print(f" Model Accuracy: {(evaluater.accuracy(validation_loader) * 100):.4f} %")

    evaluater.evaluate(num_images=5)



if __name__ == "__main__":
    main()
