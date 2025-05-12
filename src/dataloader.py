import os
import glob
import json

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

root_dir = "./TestDataSet"
mean_norms = [0.485, 0.456, 0.406]
std_norms = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norms, std=std_norms)
])

with open(os.path.join(root_dir, "labels_list.json"), "r") as f:
    idx_to_label = json.load(f)

image_label_list = []
for imagenet_idx, wnid in enumerate(sorted(os.listdir(root_dir))):
    full_path = os.path.join(root_dir, wnid)
    if not os.path.isdir(full_path) or wnid.startswith("."):
        continue
    image_files = glob.glob(os.path.join(full_path, "*.JPEG"))
    for img_path in image_files:
        image_label_list.append((img_path, imagenet_idx + 398))

class CustomImageNetDataset(Dataset):
    def __init__(self, image_label_list, transform=None):
        self.data = image_label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = CustomImageNetDataset(image_label_list, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(len(dataset))
print(len(loader))

import torch.utils.data as data

class AdversarialDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

adv_dataset = AdversarialDataset(adv_images, adv_labels)
adv_loader = DataLoader(adv_dataset, batch_size=32, shuffle=False)

pgd_adv_dataset = AdversarialDataset(pgd_adv_images, pgd_adv_labels)
pgd_adv_loader = DataLoader(pgd_adv_dataset, batch_size=32, shuffle=False)

patch_adv_dataset = AdversarialDataset(patch_adv_images, patch_adv_labels)
patch_adv_loader = DataLoader(patch_adv_dataset, batch_size=32, shuffle=False)

top1_total = 0
top5_total = 0
n_samples = 0

with torch.no_grad():
    for images, labels in tqdm(patch_adv_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.topk(5, 1, True, True)

        top1_preds = preds[:, 0]
        top5_correct = (preds == labels.view(-1, 1)).any(dim=1)

        top1_total += (top1_preds == labels).sum().item()
        top5_total += top5_correct.sum().item()
        n_samples += labels.size(0)

print(f"Patch Attack Top-1 Accuracy: {top1_total / n_samples:.4f}")
print(f"Patch Attack Top-5 Accuracy: {top5_total / n_samples:.4f}")

def evaluate_model(model, dataloader):
    top1_total = 0
    top5_total = 0
    n_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.topk(5, 1, True, True)

            top1_preds = preds[:, 0]
            top5_correct = (preds == labels.view(-1, 1)).any(dim=1)

            top1_total += (top1_preds == labels).sum().item()
            top5_total += top5_correct.sum().item()
            n_samples += labels.size(0)

    top1_acc = top1_total / n_samples
    top5_acc = top5_total / n_samples
    return top1_acc, top5_acc