# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Assignment 02 - Logistic Regression Classifier using PyTorch on CIFAR-10 Dataset
#
# #### Submitted By:
# 1. Dhruvan Ganesh
# 2. Sheikh Mastura Farzana
#
# %% [markdown]
# #### Importing modules and dataset

# %%
get_ipython().run_line_magic("matplotlib", "inline")

get_ipython().run_line_magic("load_ext", "nb_black")

# %%

import torch
from torch import nn
from torchvision import transforms, datasets  # import itertools

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "~/.datasets"

print("Device:", device)
print("Data Dir:", data_dir)


# %%
class Trainer(object):
    """
    The Trainer class,
    which makes use of Torch's Module, Loss, Optimizer implementations
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
    ):
        self._model = model
        self._loss = loss
        self._optimizer = optimizer

    def __str__(self):
        return f"""Trainer:\nArch:\n{self._model}\nLoss: {self._loss}"""

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs=100,
        print_log=True,
        log_interval=5,
    ):
        for epoch in range(1, epochs + 1):
            self._model.train()
            loss = None
            for batch_id, (X, Y) in enumerate(dataloader):
                X, Y = X.to(device), Y.to(device)
                X = X.reshape(dataloader.batch_size, -1)

                self._optimizer.zero_grad()
                predictions = self._model(X)
                loss = self._loss(predictions, Y)
                loss.backward()
                self._optimizer.step()

            if print_log and epoch % log_interval == 0:
                print("Epoch: {}\tLoss: {:.6f}".format(epoch, loss,))


# %%
class CIFAR10Data:
    def __init__(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        self.testset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )

        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def get_cifar10_data(self):
        return self.trainset, self.testset, self.classes

    def get_cifar10_batch_loaders(self, batch_size=64):
        train_dataloader = torch.utils.data.DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=2,
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.testset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            num_workers=2,
        )

        return train_dataloader, test_dataloader


# %% [markdown]
# #### Calculating Loss

# %%
cifar10_data = CIFAR10Data()
batch_size = 64

train_dataloader, test_dataloader = cifar10_data.get_cifar10_batch_loaders(
    batch_size=batch_size
)

# %%
model = nn.Sequential(
    nn.Linear(1024 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(1),
).to(device)

trainer = Trainer(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
)

print(trainer)

# %%
trainer.fit(train_dataloader)

# %% [markdown]
# #### Testing with the best parameters
# %% [markdown]
# #### Testing with the best parameters
