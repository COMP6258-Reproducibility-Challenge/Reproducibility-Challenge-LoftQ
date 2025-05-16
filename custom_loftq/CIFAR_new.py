import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, Bottleneck
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm

# --- Stochastic Depth ---

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

class BottleneckSD(Bottleneck):
    def __init__(self, *args, drop_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path = StochasticDepth(drop_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet50_stochastic_depth(drop_prob=0.05, num_classes=10):
    layers = [3, 4, 6, 3]
    model = ResNet(BottleneckSD, layers, num_classes=num_classes)
    total_blocks = sum(layers)
    block_idx = 0
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            block.drop_path.drop_prob = drop_prob * float(block_idx) / total_blocks
            block_idx += 1
    return model

# --- Dataset with repeated augmentation ---

class CIFAR10OneHotRepeated(Dataset):
    def __init__(self, train=True, repeats=3, transform=None):
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        self.repeats = repeats
        self.transform = transform
        self.num_samples = len(self.dataset) * self.repeats

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        real_idx = idx % len(self.dataset)
        image, label = self.dataset[real_idx]
        if self.transform:
            image = self.transform(image)
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0
        return {'pixel_values': image, 'labels': label_onehot}

# --- Mixup and CutMix collator ---

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_cutmix_collator(batch, alpha_mixup=0.2, alpha_cutmix=1.0):
    images = torch.stack([x['pixel_values'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    batch_size = images.size(0)

    if batch_size < 2:
        # Can't mixup or cutmix with batch size 1
        return {'pixel_values': images, 'labels': labels}

    use_cutmix = random.random() > 0.5

    if use_cutmix:
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        rand_index = torch.randperm(batch_size)
        images_clone = images.clone()
        images[:, :, bbx1:bbx2, bby1:bby2] = images_clone[rand_index, :, bbx1:bbx2, bby1:bby2]
        labels = lam * labels + (1 - lam) * labels[rand_index]
    else:
        lam = np.random.beta(alpha_mixup, alpha_mixup)
        rand_index = torch.randperm(batch_size)
        images = lam * images + (1 - lam) * images[rand_index]
        labels = lam * labels + (1 - lam) * labels[rand_index]

    return {'pixel_values': images, 'labels': labels}


if __name__ == '__main__':
    # --- Transforms ---

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=7, magnitude=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    # --- DataLoaders ---

    train_dataset = CIFAR10OneHotRepeated(train=True, repeats=3, transform=train_transform)
    val_dataset = CIFAR10OneHotRepeated(train=False, repeats=1, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=mixup_cutmix_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model, optimizer, scheduler ---

    model = resnet50_stochastic_depth(drop_prob=0.05, num_classes=10).to(device)

    optimizer = Lamb(model.parameters(), lr=5e-3, weight_decay=0.01)

    total_epochs = 600
    warmup_epochs = 5
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        warmup_t=warmup_steps,
        cycle_limit=1,
    )

    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()

    # --- Training loop ---

    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", ncols=100)
        for step, batch in enumerate(train_bar):
            images = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(epoch * steps_per_epoch + step)

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (step + 1))

        # Validation
        model.eval()
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", ncols=100)
        with torch.no_grad():
            for batch in val_bar:
                images = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
                total += labels.size(0)
                val_bar.set_postfix(accuracy=correct / total)

        acc = correct / total
        print(f"Epoch {epoch+1}/{total_epochs} Validation accuracy: {acc:.4f}")
