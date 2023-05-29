import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm


np.random.seed(0)
torch.manual_seed(0)

# Training hyperparameters
BATCH_SIZE = 16
xi_to_ix_dir = os.path.join("..", "xi_to_ix")


def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def create_xi_to_ix_photos(xi_to_ix_dir):
    mirror_transforms = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
        ])

    transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image
])
    
    xi_to_ix_dataset = datasets.ImageFolder(xi_to_ix_dir, mirror_transforms)
    ix_dataloader = torch.utils.data.DataLoader(xi_to_ix_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for i, batch in enumerate(ix_dataloader):
        images, _ = batch  # Unpack the batch into images and labels

        # Iterate over the images in the batch
        for j, image in enumerate(images):
            # Apply transformations
            image = image.squeeze()  # Remove the batch dimension
            print(image.shape)
            image_pil = transform(image)

            # Save the image
            save_path = os.path.join(xi_to_ix_dir, f"image_{i}_{j}.jpg")
            image_pil.save(save_path)


create_xi_to_ix_photos(xi_to_ix_dir)


