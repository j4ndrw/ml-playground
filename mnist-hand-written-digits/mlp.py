# cba to add monitoring. i know wandb and others are a thing, but im too lazy to set it up.
# good enough as it is!

import torch
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor


class MLPDigitClassifier(nn.Module):
    def __init__(self, rows: int, cols: int):
        super().__init__()

        self.input_proj = nn.Linear(rows * cols, 2048)
        self.fc = nn.Linear(2048, 2048)
        self.classes = nn.Linear(2048, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.flatten(x, 2)
        out = self.input_proj.forward(out)
        out = self.fc.forward(out)
        out = self.classes.forward(out)
        out = F.relu(out)
        return out

    def inference(self, x: torch.Tensor) -> int:
        return int(self.forward(x).argmax(dim=-1).view(x.size(0)).contiguous().item())

    def fit(
        self,
        *,
        epochs: int,
        log_every: int,
        lr: float,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        optimizer = SGD(self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            self.train()
            total_training_loss = 0.0
            for _batch in train_dataloader:
                with torch.enable_grad():
                    optimizer.zero_grad()
                    batch: tuple[torch.Tensor, torch.Tensor] = _batch
                    inputs, labels = batch
                    labels = (
                        F.one_hot(labels, num_classes=10).to(torch.float).argmax(dim=-1)
                    )

                    prediction = self.forward(inputs).squeeze()
                    loss = F.cross_entropy(prediction, labels)
                    loss.backward()

                    optimizer.step()

                    total_training_loss += loss.item()

            if epoch % log_every == 0:
                print(
                    f"{epoch=} | avg_training_loss={total_training_loss/len(train_dataloader)} | ",
                    end="",
                )

            self.eval()
            total_validation_loss = 0.0
            for _batch in test_dataloader:
                batch: tuple[torch.Tensor, torch.Tensor] = _batch
                inputs, labels = batch
                labels = (
                    F.one_hot(labels, num_classes=10).to(torch.float).argmax(dim=-1)
                )

                prediction = self.forward(inputs).squeeze()
                loss = F.cross_entropy(prediction, labels)

                total_validation_loss += loss.item()

            if epoch % log_every == 0:
                print(
                    f"avg_validation_loss={total_validation_loss/len(test_dataloader)}"
                )


if __name__ == "__main__":
    batch_size = 4

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloader = DataLoader(
        Subset(training_data, list(range(2048))), batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        Subset(test_data, list(range(128))), batch_size=batch_size, shuffle=True
    )
    inference_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = MLPDigitClassifier(28, 28)
    model.fit(
        epochs=100,
        log_every=1,
        lr=1e-3,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    print("\n----------------------------------------------------------------\n")
    print("\tINFERENCE")
    rights = 0
    already_inferred: set[int] = set()
    for _batch in inference_dataloader:
        batch: tuple[torch.Tensor, torch.Tensor] = _batch
        inputs, labels = batch
        label = int(labels.squeeze().item())
        prediction = model.inference(inputs)
        rights += label == prediction
        if label in already_inferred:
            continue
        already_inferred.add(label)
        print(f"\t\tExpected {label}, found {prediction}")
    print(
        f"[\t\t\t\tMODEL IS RIGHT %{rights * 100 / len(inference_dataloader):.2f}% OF THE TIME\t\t\t\t]"
    )
