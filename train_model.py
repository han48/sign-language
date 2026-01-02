import os
import pickle
import json
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import csv
import math
import random
import glob
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings('ignore')

NUM_CLASSES = 100
TARGET_FRAMES = 16

def read_video(video_path):
    """Read video frames using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    frames = torch.from_numpy(np.stack(frames, axis=0))
    return frames


def collate_fn(batch):
    """Custom collate function for batching"""
    frames = torch.stack([item['frames'] for item in batch])
    labels = torch.tensor([item['label_idx'] for item in batch])
    label_names = [item['label'] for item in batch]
    return {'frames': frames, 'label_idx': labels, 'label': label_names}


class VideoAugmentation:
    """
    Augmentation cho video - CONSISTENT across all frames
    Chỉ dùng cho training, không dùng cho val/test
    """
    def __init__(self,
                 crop_scale=(0.85, 1.0),
                 brightness=0.2,
                 contrast=0.2,
                 saturation=0.2,
                 speed_range=(0.9, 1.1)):

        self.crop_scale = crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.speed_range = speed_range

    def __call__(self, frames):
        """
        Args:
            frames: tensor (T, H, W, C) với giá trị 0-255
        Returns:
            augmented frames: tensor (T, H, W, C)
        """
        # 1. Speed Augmentation (thay đổi số frames)
        frames = self._speed_augment(frames)

        # 2. Random Resized Crop (CONSISTENT cho tất cả frames)
        frames = self._random_resized_crop(frames)

        # 3. Color Jitter (CONSISTENT cho tất cả frames)
        frames = self._color_jitter(frames)

        return frames

    def _speed_augment(self, frames):
        """Thay đổi tốc độ video bằng cách resample frames"""
        T = frames.shape[0]
        speed = random.uniform(self.speed_range[0], self.speed_range[1])

        new_T = int(T / speed)
        if new_T < 4:
            new_T = 4
        if new_T == T:
            return frames

        # Resample frames
        indices = torch.linspace(0, T - 1, new_T).long()
        indices = torch.clamp(indices, 0, T - 1)
        frames = frames[indices]

        return frames

    def _random_resized_crop(self, frames):
        """Random crop rồi resize về 224x224 - CONSISTENT"""
        T, H, W, C = frames.shape

        # Random scale và position (CÙNG cho tất cả frames)
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_h, crop_w = int(H * scale), int(W * scale)

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # Crop tất cả frames GIỐNG NHAU
        frames = frames[:, top:top+crop_h, left:left+crop_w, :]

        # Resize về 224x224
        # (T, H, W, C) -> (T, C, H, W) for interpolate
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        # (T, C, H, W) -> (T, H, W, C)
        frames = frames.permute(0, 2, 3, 1)

        return frames.to(torch.uint8)

    def _color_jitter(self, frames):
        """Color jitter - CONSISTENT cho tất cả frames"""
        # Random parameters (CÙNG cho tất cả frames)
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        frames = frames.float()

        # Brightness
        frames = frames * brightness_factor

        # Contrast
        mean = frames.mean(dim=(1, 2), keepdim=True)
        frames = (frames - mean) * contrast_factor + mean

        # Saturation
        gray = frames.mean(dim=-1, keepdim=True)
        frames = gray + (frames - gray) * saturation_factor

        # Clamp to valid range
        frames = torch.clamp(frames, 0, 255)

        return frames.to(torch.uint8)


class VideoDataset(Dataset):
    def __init__(self, root_dir, label_to_idx_path, transform=None,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 target_frames=16, training=False):

        self.root_dir = root_dir
        self.transform = transform
        self.mean, self.std = mean, std
        self.target_frames = target_frames
        self.training = training

        # Augmentation chỉ khi training
        self.augmentation = VideoAugmentation() if training else None

        self.instances, self.labels, self.label_idx = [], [], []

        with open(label_to_idx_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        # Normalize Unicode keys to NFC form
        self.label_mapping = {unicodedata.normalize('NFC', k): v for k, v in self.label_mapping.items()}

        for label_folder in sorted(os.listdir(root_dir))[:NUM_CLASSES]:
            path = os.path.join(root_dir, label_folder)
            if os.path.isdir(path):
                for video_file in os.listdir(path):
                    video_path = os.path.join(path, video_file)
                    self.instances.append(video_path)
                    self.labels.append(label_folder)
                    # Normalize label_folder to NFC form before looking up in label_mapping
                    self.label_idx.append(self.label_mapping[unicodedata.normalize('NFC', label_folder)])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        video_path = self.instances[idx]
        frames = read_video(video_path)

        # ============ AUGMENTATION (chỉ khi training) ============
        if self.training and self.augmentation is not None:
            frames = self.augmentation(frames)
        # ========================================================

        frames = self._downsample_frames(frames)
        frames = self._normalize(frames)

        return {
            'frames': frames,
            'label_idx': self.label_idx[idx],
            'label': self.labels[idx]
        }

    def _downsample_frames(self, frames):
        """Lấy target_frames từ video"""
        total = frames.shape[0]
        if total >= self.target_frames:
            indices = torch.linspace(0, total - 1, self.target_frames).long()
        else:
            indices = torch.arange(total)
            pad = self.target_frames - total
            indices = torch.cat([indices, indices[-1].repeat(pad)])

        frames = frames[indices]

        # Resize về 224x224 nếu chưa
        if frames.shape[1] != 224 or frames.shape[2] != 224:
            frames = frames.permute(0, 3, 1, 2).float()
            frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
            frames = frames.permute(0, 2, 3, 1).to(torch.uint8)

        return frames

    def _normalize(self, frames):
        """Normalize về ImageNet mean/std"""
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        return frames


def create_balanced_sampler(dataset):
    """Create balanced sampler for imbalanced dataset"""
    if hasattr(dataset, 'dataset'):
        all_labels = [dataset.dataset.label_idx[i] for i in dataset.indices]
    else:
        all_labels = dataset.label_idx

    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in all_labels]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Balanced Sampler: class counts min={class_counts.min()}, max={class_counts.max()}")
    return sampler


class PositionalEncoding(nn.Module):
    """Positional encoding cho temporal sequence"""
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention pooling thay cho last hidden của LSTM"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: (B, T, dim)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, dim)
        return pooled


class ConvNeXtTransformer(nn.Module):
    """
    ConvNeXt-Tiny + Transformer

    Input:  (B, T, C, H, W) = (B, 16, 3, 224, 224)
    Output: (B, num_classes) = (B, 100)
    """
    def __init__(self, num_classes=100, hidden_size=256, resnet_pretrained_weights=None):
        super().__init__()

        # 1. ConvNeXt-Tiny Backbone
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.cnn = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ConvNeXt-Tiny output = 768
        self.feature_dim = 768

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.feature_dim,
            max_len=64,
            dropout=0.1
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Attention Pooling
        self.attention_pool = AttentionPooling(self.feature_dim)

        # 5. Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # CNN: (B, T, C, H, W) → (B, T, 768)
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(B, T, self.feature_dim)

        # Transformer: (B, T, 768) → (B, T, 768)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Pooling: (B, T, 768) → (B, 768)
        x = self.attention_pool(x)

        # Classifier: (B, 768) → (B, num_classes)
        x = self.fc(x)

        return x


def evaluate(model, folder_path, label_to_idx_path, output_csv="predictions.csv",
             device='cuda', model_path=None, target_frames=16):
    """Evaluate trained model on test set"""
    # Load trained weights if provided
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    model = model.to(device)
    model.eval()

    # Load label mapping
    with open(label_to_idx_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    # Normalize Unicode keys to NFC form
    label_mapping = {v: k for k, v in label_mapping.items()}
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Collect video files
    video_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    print(f"Found {len(video_files)} videos in '{folder_path}'")

    predictions = []

    dataset = VideoDataset(
        root_dir=folder_path,
        label_to_idx_path=label_to_idx_path,
        target_frames=target_frames
    )

    with torch.no_grad():
        for video_file in tqdm(video_files, desc="Predicting"):
            video_path = os.path.join(folder_path, video_file)
            try:
                # Read and preprocess video
                frames = read_video(video_path)
                frames = dataset._downsample_frames(frames)
                frames = dataset._normalize(frames)
                frames = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)

                # Predict
                outputs = model(frames)
                _, predicted = outputs.max(1)
                label_idx = predicted.item()
                label_name = idx_to_label[label_idx]

                predictions.append((video_file, label_name))
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    # Save to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'label'])
        writer.writerows(predictions)

    print(f"\nPredictions saved to '{output_csv}'")
    print(f"Total videos processed: {len(predictions)}")


def train_epoch(model, dataloader, criterion, optimizer, device='cuda'):
    """One training epoch"""
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc='Training')
    for batch in progress:
        frames, labels = batch['frames'].to(device), batch['label_idx'].to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{total_loss / (len(progress)+1e-9):.4f}'})
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device='cuda'):
    """Validation"""
    model.eval()
    total_loss, preds, labels_all = 0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            frames, labels = batch['frames'].to(device), batch['label_idx'].to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average='macro', zero_division=0)
    return total_loss / len(dataloader), {'precision': precision*100, 'recall': recall*100, 'f1': f1*100}


def cleanup_old_checkpoints(max_checkpoints):
    """Remove old checkpoints, keeping only the latest max_checkpoints"""
    if max_checkpoints is None or max_checkpoints <= 0:
        return
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_files = glob.glob('checkpoints/checkpoint_epoch_*.pth')
    if len(checkpoint_files) <= max_checkpoints:
        return
    
    # Extract epoch numbers
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_')[-1].replace('.pth', ''))
            epochs.append((epoch, f))
        except ValueError:
            continue
    
    # Sort by epoch descending (newest first)
    epochs.sort(key=lambda x: x[0], reverse=True)
    
    # Keep only max_checkpoints, remove the rest
    to_remove = epochs[max_checkpoints:]
    for _, f in to_remove:
        os.remove(f)
        print(f"Removed old checkpoint: {f}")


def train_model(model, train_loader, val_loader,
                num_epochs=20, lr=1e-4, device='cuda', save_path='best_model.pth', resume_epoch=None, max_checkpoints=None):
    """Full training loop with validation and test evaluation"""
    model = model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_f1 = 0.0
    if resume_epoch is not None:
        checkpoint_path = f'checkpoints/checkpoint_epoch_{resume_epoch}.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # If optimizer state is saved, load it too (optional)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_f1 = checkpoint.get('best_f1', 0.0)
            print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.2f}%")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from epoch 0")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3
    )

    # Check if results.csv exists, if not, write header
    results_file = 'results.csv'
    file_exists = os.path.exists(results_file)
    with open(results_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'f1'])

    for epoch in range(start_epoch, num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} ======")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Val F1: {val_metrics['f1']:.2f}% | Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}%")

        # Write to CSV
        with open(results_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_metrics['precision']:.2f}",
                f"{val_metrics['recall']:.2f}",
                f"{val_metrics['f1']:.2f}"
            ])

        # Always save checkpoint with epoch index
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1
        }
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(max_checkpoints)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # Also save as best_model.pth
            torch.save(checkpoint, save_path)
            print(f"✓ Best model saved with F1: {best_f1:.2f}%")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Video Model')
    parser.add_argument('--resume', type=int, default=None, help='Epoch number to resume training from (loads checkpoints/checkpoint_epoch_{epoch}.pth)')
    parser.add_argument('--max-checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep (default: keep all)')
    args = parser.parse_args()

    # Tạo datasets
    train_dataset_base = VideoDataset(
        'dataset/train',
        'dataset/label_mapping.json',
        target_frames=TARGET_FRAMES,
        training=True  # CÓ augmentation
    )

    val_dataset_base = VideoDataset(
        'dataset/train',
        'dataset/label_mapping.json',
        target_frames=TARGET_FRAMES,
        training=False  # KHÔNG augmentation
    )

    # Split
    train_size = int(0.8 * len(train_dataset_base))
    val_size = len(train_dataset_base) - train_size

    indices = list(range(len(train_dataset_base)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset_base, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)

    balanced_sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=balanced_sampler,  # ← THAY shuffle=True
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print(f"Train: {len(train_dataset)} (augmentation + balanced sampling)")
    print(f"Val: {len(val_dataset)} (no augmentation)")

    model = ConvNeXtTransformer(num_classes=NUM_CLASSES, hidden_size=256,
                 resnet_pretrained_weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=25,
        lr=1e-4,
        device='cuda',
        save_path='augmented_balanced_convnexttransformer_best_model.pth',
        resume_epoch=args.resume,
        max_checkpoints=args.max_checkpoints
    )

    # Export public result
    evaluate(
        model=model,
        folder_path="dataset/public_test",
        label_to_idx_path="dataset/label_mapping.json",
        model_path="augmented_balanced_convnexttransformer_best_model.pth",
        output_csv="public_test.csv",
        device="cuda",
        target_frames=16
    )

    # Export public result
    evaluate(
        model=model,
        folder_path="dataset/private_test",
        label_to_idx_path="dataset/label_mapping.json",
        model_path="augmented_balanced_convnexttransformer_best_model.pth",
        output_csv="private_test.csv",
        device="cuda",
        target_frames=16
    )