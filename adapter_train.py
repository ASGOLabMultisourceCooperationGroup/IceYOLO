import os

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ultralytics.nn.modules import Adapter

print("Modules loaded")

# Change these settings
NAME = "yrcc1"
INPUT_CHANNEL = 3
OUTPUT_CHANNEL = 16
PATH = "/data/home/scv8591/run/yuka/datasets/dataset_yrcc1/images"
BATCH = 32
IMG_SIZE = 768
EPOCH = 100
NORMALIZE = 255


class ImageDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.images = os.listdir(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.images[idx])
        image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

        H, W, C = image.shape
        assert H == W == IMG_SIZE and C == INPUT_CHANNEL

        # im = image[:, :, [1, 2, 3]]
        # cv2.imshow("Original", im)
        # cv2.waitKey(0)
        image = torch.from_numpy(image).permute(2, 0, 1).float().to("cuda")
        image /= NORMALIZE
        return image


print("Start loading data...")
dataset = ImageDataset(PATH)
num_train = int(len(dataset) * 0.8)
num_val = len(dataset) - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True)
print("Data loaded.")


def train():
    model = Adapter(INPUT_CHANNEL).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                           threshold=0.01)
    loss_func = torch.nn.L1Loss()
    epoch_min_loss = float('inf')
    for epoch in range(EPOCH):
        model.train()
        model.training = True
        batch_count = 0
        epoch_loss = float(0)
        for batch_index, img_input in enumerate(train_dataloader):
            # for im in img_input:
            #     im = im.permute(1, 2, 0).cpu().numpy()
            #     im = im[:, :, [3, 2, 1]]
            #     cv2.imshow("Unfolded", im)
            #     cv2.waitKey(0)
            batch_count += 1
            optimizer.zero_grad()
            unfolded = model(img_input)
            loss = loss_func(unfolded, img_input)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f"\rEpoch {epoch}, Batch {batch_index}, Loss {loss.item()}", end='')
            torch.cuda.empty_cache()
        scheduler.step(epoch_loss / batch_count)
        if epoch_loss / batch_count < epoch_min_loss:
            epoch_min_loss = epoch_loss / batch_count
            torch.save(model.state_dict(), f"weights/adapter_{NAME}_best.pth")
        print(f"\nEpoch {epoch} finished. Loss {epoch_loss / batch_count}")
        val_onflight(model)
        if epoch_loss / batch_count < 0.005:
            break

    print("Saving model...")
    torch.save(model.state_dict(), f"weights/adapter_{NAME}_last.pth")
    print("Training finished.")


def val_onflight(model):
    model.eval()
    model.training = True
    total_loss = float(0)
    batch_count = 0
    loss_func = torch.nn.L1Loss()
    for batch_index, img_input in enumerate(val_dataloader):
        with torch.no_grad():
            unfolded = model(img_input)
            loss = loss_func(unfolded, img_input).item()
            # for im in unfolded:
            #     im = im.permute(1, 2, 0).cpu().numpy()
            #     im = im[:, :, [1, 2, 3]]
            #     cv2.imshow("Unfolded", im)
            #     cv2.waitKey(0)
        total_loss += loss
        batch_count += 1
        print(f"\rBatch {batch_index} Loss {loss}", end='')
        torch.cuda.empty_cache()
    print(f"\nValidation loss {total_loss / batch_count}")


def val():
    model = Adapter(INPUT_CHANNEL).to("cuda")
    # model.load_state_dict(torch.load(f"{PROJECT_NAME}_best.pth"))
    model.load_state_dict(torch.load("sa_best_bak.pth"))
    model.eval()
    total_loss = float(0)
    batch_count = 0
    loss_func = torch.nn.L1Loss()
    for batch_index, img_input in enumerate(val_dataloader):
        with torch.no_grad():
            unfolded = model(img_input)
            loss = loss_func(unfolded, img_input).item()
            for im in unfolded:
                im = im.permute(1, 2, 0).cpu().numpy()
                im = im[:, :, [1, 2, 3]]
                cv2.imshow("Unfolded", im)
                cv2.waitKey(0)
        total_loss += loss
        batch_count += 1
        print(f"\rBatch {batch_index} Loss {loss}", end='')
        torch.cuda.empty_cache()
    print(f"\nValidation loss {total_loss / batch_count}")


train()
# val()
