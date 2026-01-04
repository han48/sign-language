from time import time
import os
import cv2
import csv
import json
import math
import glob
import torch
import random
import numpy as np
import unicodedata
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms as T
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

TARGET_FRAMES = 16


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
        frames = F.interpolate(frames, size=(224, 224),
                               mode='bilinear', align_corners=False)
        # (T, C, H, W) -> (T, H, W, C)
        frames = frames.permute(0, 2, 3, 1)

        return frames.to(torch.uint8)

    def _color_jitter(self, frames):
        """Color jitter - CONSISTENT cho tất cả frames"""
        # Random parameters (CÙNG cho tất cả frames)
        brightness_factor = 1.0 + \
            random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + \
            random.uniform(-self.saturation, self.saturation)

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
    def __init__(self, model, root_dir, label_to_idx_path="dataset/label_mapping.json", transform=None,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 target_frames=16, training=False):

        self.model = model
        self.root_dir = root_dir
        self.label_to_idx_path = label_to_idx_path
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
        self.label_mapping = {unicodedata.normalize(
            'NFC', k): v for k, v in self.label_mapping.items()}

        for label_folder in sorted(os.listdir(root_dir))[:len(self.label_mapping)]:
            path = os.path.join(root_dir, label_folder)
            if os.path.isdir(path):
                for video_file in os.listdir(path):
                    video_path = os.path.join(path, video_file)
                    self.instances.append(video_path)
                    self.labels.append(label_folder)
                    # Normalize label_folder to NFC form before looking up in label_mapping
                    self.label_idx.append(
                        self.label_mapping[unicodedata.normalize('NFC', label_folder)])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        video_path = self.instances[idx]
        frames, _ = self.model.read_video(video_path, update_bar=False)

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
            frames = F.interpolate(frames, size=(
                224, 224), mode='bilinear', align_corners=False)
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

    def create_balanced_sampler(self, dataset):
        """Create balanced sampler for imbalanced dataset"""
        if hasattr(dataset, 'dataset'):
            all_labels = [dataset.dataset.label_idx[i]
                          for i in dataset.indices]
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

        print(
            f"Balanced Sampler: class counts min={class_counts.min()}, max={class_counts.max()}")
        return sampler

    def collate_fn(self, batch):
        """Custom collate function for batching"""
        frames = torch.stack([item['frames'] for item in batch])
        labels = torch.tensor([item['label_idx'] for item in batch])
        label_names = [item['label'] for item in batch]
        return {'frames': frames, 'label_idx': labels, 'label': label_names}


class PositionalEncoding(nn.Module):
    """Positional encoding cho temporal sequence"""

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

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


class VideoPreprocessor:
    def __init__(self, model, target_frames=32, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.target_frames = target_frames
        self.mean = mean
        self.std = std
        self.model = model

    def _downsample_frames(self, frames):
        num_frames = frames.shape[0]
        if num_frames == self.target_frames:
            return frames
        elif num_frames < self.target_frames:
            pad = self.target_frames - num_frames
            return torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
        else:
            idx = torch.linspace(0, num_frames - 1, self.target_frames).long()
            return frames[idx]

    def _normalize(self, frames):
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        # Center crop to square to avoid distortion
        _, _, H, W = frames.shape
        min_side = min(H, W)
        crop = T.CenterCrop(min_side)
        frames = crop(frames)
        # Resize to 224x224
        resize = T.Resize((224, 224))
        frames = resize(frames)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (frames - mean) / std

    def preprocess(self, video_path):
        frames = self.model.read_video(video_path, update_bar=False)
        frames = self._downsample_frames(frames)
        frames = self._normalize(frames)
        return frames.unsqueeze(0)  # Add batch dimension


class ConvNeXtTransformer(nn.Module):
    """
    ConvNeXt-Tiny + Transformer

    Input:  (B, T, C, H, W) = (B, 16, 3, 224, 224)
    Output: (B, num_classes) = (B, 100)
    """

    def __init__(self, num_classes=100, hidden_size=256, resnet_pretrained_weights=None):
        super().__init__()

        # 1. ConvNeXt-Tiny Backbone
        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
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

    def read_video(self, video_path, update_bar=True, fn_push=None):
        """Read video frames using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0:
            fps = 30  # Default fallback
        frames = []
        if update_bar:
            with tqdm(desc="Reading video frames", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if fn_push is not None:
                        fn_push(["Reading video frames", pbar.n, total_frames])
        else:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if fn_push is not None:
                    frame_index += 1
                    fn_push(["Reading video frames", frame_index, total_frames])
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from {video_path}")
        frames = torch.from_numpy(np.stack(frames, axis=0))
        return frames, fps

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

    def evaluate(self, folder_path, label_to_idx_path, output_csv="predictions.csv",
                 device='cuda', model_path=None, target_frames=16, save_directory=""):
        """Evaluate trained model on test set"""
        # Load trained weights if provided
        if model_path:
            model = ConvNeXtTransformer.load_model(model_path, device)
        else:
            model = self.to(device)

        # Load label mapping
        with open(label_to_idx_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        # Normalize Unicode keys to NFC form
        label_mapping = {v: k for k, v in label_mapping.items()}
        idx_to_label = {v: k for k, v in label_mapping.items()}

        # Collect video files
        video_files = sorted([f for f in os.listdir(
            folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        print(f"Found {len(video_files)} videos in '{folder_path}'")

        predictions = []

        dataset = VideoDataset(
            model,
            folder_path,
            label_to_idx_path=label_to_idx_path,
            target_frames=target_frames
        )

        with torch.no_grad():
            for video_file in tqdm(video_files, desc="Predicting"):
                video_path = os.path.join(folder_path, video_file)
                try:
                    # Read and preprocess video
                    frames, _ = self.read_video(video_path, update_bar=False)
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
        with open(f"{save_directory}/{output_csv}", mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_name', 'label'])
            writer.writerows(predictions)

        print(f"\nPredictions saved to '{output_csv}'")
        print(f"Total videos processed: {len(predictions)}")

    def train_epoch(self, dataloader, criterion, optimizer, device='cuda'):
        """One training epoch"""
        model = self.to(device)
        model.eval()

        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc='Training')
        for batch in progress:
            frames, labels = batch['frames'].to(
                device), batch['label_idx'].to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(
                {'loss': f'{total_loss / (len(progress)+1e-9):.4f}'})
        return total_loss / len(dataloader)

    def validate(self, dataloader, criterion, device='cuda'):
        """Validation"""
        model = self.to(device)
        model.eval()

        total_loss, preds, labels_all = 0, [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                frames, labels = batch['frames'].to(
                    device), batch['label_idx'].to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_all, preds, average='macro', zero_division=0)
        return total_loss / len(dataloader), {'precision': precision*100, 'recall': recall*100, 'f1': f1*100}

    def cleanup_old_checkpoints(self, max_checkpoints, save_directory=""):
        """Remove old checkpoints, keeping only the latest max_checkpoints"""
        if max_checkpoints is None or max_checkpoints <= 0:
            return

        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_files = glob.glob(
            f'{save_directory}/checkpoints/checkpoint_epoch_*.pth')
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

    def train_model(self, train_loader, val_loader,
                    num_epochs=20, lr=1e-4, device='cuda', save_path='best_model.pth', label_mapping_path='dataset/label_mapping.json', resume_epoch=None, max_checkpoints=None, save_directory=""):
        """Full training loop with validation and test evaluation"""
        model = self.to(device)
        model.eval()

        label_mapping_file = label_mapping_path
        label_mapping = {}
        if label_mapping_file.endswith('.json'):
            with open(label_mapping_file, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)

        # Resume from checkpoint if provided
        start_epoch = 0
        best_f1 = 0.0
        if resume_epoch is not None:
            checkpoint_path = f'{save_directory}/checkpoints/checkpoint_epoch_{resume_epoch}.pth'
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                # If optimizer state is saved, load it too (optional)
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_f1 = checkpoint.get('best_f1', 0.0)
                print(
                    f"Resumed from epoch {start_epoch}, best F1: {best_f1:.2f}%")
            else:
                print(
                    f"Checkpoint {checkpoint_path} not found, starting from epoch 0")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=3
        )

        # Check if results.csv exists, if not, write header
        results_file = f'{save_directory}/results.csv'
        file_exists = os.path.exists(results_file)
        with open(results_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss',
                                'precision', 'recall', 'f1'])

        for epoch in range(start_epoch, num_epochs):
            print(f"\n===== Epoch {epoch+1}/{num_epochs} ======")
            train_loss = self.train_epoch(
                train_loader, criterion, optimizer, device)
            val_loss, val_metrics = self.validate(
                val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Val F1: {val_metrics['f1']:.2f}% | Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}%")

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
            os.makedirs(f'{save_directory}/checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': (epoch + 1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': max(best_f1, val_metrics['f1']),
                'f1': val_metrics['f1'],
                'recall': val_metrics['recall'],
                'precision': val_metrics['precision'],
                'idx_to_label': {v: k for k, v in label_mapping.items()},
                'predict_steps': [
                    "Reading video frames",
                    "Processing video windows",
                    "Processing temporal blocks",
                ]
            }
            checkpoint_path = f'{save_directory}/checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            self.cleanup_old_checkpoints(
                max_checkpoints, save_directory=save_directory)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                # Also save as best_model.pth
                torch.save(checkpoint, f"{save_directory}/{save_path}")
                print(f"✓ Best model saved with F1: {best_f1:.2f}%")

        return model

    def predict_sign_language_sentence(self, video_path, fn_push=None, window_size=16, stride=8, confidence_threshold=0.0, block_durations=None, target_fps=None, block_duration_for_summary=1, show=False, debug=False):

        # Read and preprocess entire video
        frames, fps = self.read_video(video_path, fn_push=fn_push)
        num_frames = frames.shape[0]
        if debug:
            print(f"Video has {num_frames} frames, FPS: {fps}")

        # Prepare display frames if show is enabled
        frames_display = None
        screen_width = None
        screen_height = None
        if show:
            cap = cv2.VideoCapture(video_path)
            frames_display = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_display.append(frame)
            cap.release()
            frames_display = np.array(frames_display)
            if debug:
                print(f"Loaded {len(frames_display)} frames for display")

        # Resample frames if target_fps is set
        if target_fps is not None and target_fps != fps:
            step = int(fps / target_fps)
            if step > 1:
                frames = frames[::step]
                num_frames = frames.shape[0]
                fps = target_fps
                if debug:
                    print(f"Resampled to {num_frames} frames at {fps} FPS")

        # Normalize frames
        preprocessor = VideoPreprocessor(self, target_frames=window_size)
        # Normalize without downsampling
        frames = preprocessor._normalize(frames)

        predictions = []

        # Sliding window prediction with progress bar
        total_windows = len(range(0, num_frames - window_size + 1, stride))
        with tqdm(total=total_windows, desc="Processing video windows") as pbar:
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                window_frames = frames[start:end]  # (window_size, C, H, W)
                window_frames = window_frames.unsqueeze(
                    0).to(model.device)  # Add batch dim

                with torch.no_grad():
                    outputs = model(window_frames)
                    probs = F.softmax(outputs, dim=1)
                    confidence, predicted = probs.max(1)
                    label_idx = predicted.item()
                    label_name = self.idx_to_label[label_idx]
                    conf_value = confidence.item()
                    predictions.append((label_name, conf_value))

                if show and frames_display is not None and end - 1 < len(frames_display):
                    display_frame = frames_display[end - 1].copy()
                    cv2.imshow('Frame', display_frame)

                    # Create a separate window for text info
                    info_image = np.zeros((100, 400, 3), dtype=np.uint8)
                    text = f"Processing frame index: {start}-{end}"
                    cv2.putText(info_image, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Info', info_image)

                    # Position windows based on screen size
                    if screen_width is not None and screen_height is not None:
                        frame_height, frame_width = display_frame.shape[:2]
                        info_x = 0
                        info_y = screen_height - 100  # Bottom-left
                        frame_x = (screen_width - frame_width) // 2
                        frame_y = (screen_height - frame_height) // 2
                        cv2.moveWindow('Info', info_x, info_y)
                        # cv2.moveWindow('Frame', frame_x, frame_y)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        show = False  # Stop showing if 'q' pressed

                pbar.set_postfix(label=f"{label_name} ({conf_value:.2f})")
                pbar.update(1)
                if fn_push is not None:
                    fn_push(["Processing video windows", pbar.n, total_windows])

        # Time-based blocks: multiple durations (1,2,3 seconds), 0.5 second shift, collect all predictions
        if block_durations is None:
            block_durations = [1, 2, 3]
        shift_duration = 0.5  # seconds
        shift_frames = int(fps * shift_duration)
        if shift_frames == 0:
            shift_frames = 15
        all_predictions = []
        total_blocks = sum((num_frames - int(fps * d) + 1) //
                           shift_frames + 1 for d in block_durations if int(fps * d) > 0)
        with tqdm(total=total_blocks, desc="Processing temporal blocks") as pbar:
            for duration in block_durations:
                frames_per_block = int(fps * duration)
                if frames_per_block == 0:
                    frames_per_block = 30 * duration  # Fallback
                for start in range(0, num_frames - frames_per_block + 1, shift_frames):
                    end = start + frames_per_block
                    start_time_sec = start / fps
                    end_time_sec = end / fps
                    window_frames = frames[start:end]
                    # Downsample to target_frames if needed
                    if window_frames.shape[0] != window_size:
                        window_frames = preprocessor._downsample_frames(
                            window_frames)
                    window_frames = window_frames.unsqueeze(0).to(model.device)
                    with torch.no_grad():
                        outputs = model(window_frames)
                        probs = F.softmax(outputs, dim=1)
                        confidence, predicted = probs.max(1)
                        label_idx = predicted.item()
                        label_name = self.idx_to_label[label_idx]
                        conf_value = confidence.item()
                        if conf_value >= confidence_threshold:
                            all_predictions.append(
                                (label_name, conf_value, start_time_sec, end_time_sec, duration))

                    if show and frames_display is not None and end - 1 < len(frames_display):
                        display_frame = frames_display[end - 1].copy()
                        cv2.imshow('Frame', display_frame)

                        # Create a separate window for text info
                        info_image = np.zeros(
                            (100, 600, 3), dtype=np.uint8)
                        text = f"Processing block: {start}-{end} (frames), {start_time_sec:.1f}-{end_time_sec:.1f}s"
                        cv2.putText(
                            info_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow('Info', info_image)

                        # Position windows based on screen size
                        if screen_width is not None and screen_height is not None:
                            frame_height, frame_width = display_frame.shape[:2]
                            info_x = 0
                            info_y = screen_height - 100  # Bottom-left
                            frame_x = (screen_width - frame_width) // 2
                            frame_y = (screen_height - frame_height) // 2
                            cv2.moveWindow('Info', info_x, info_y)
                            # cv2.moveWindow('Frame', frame_x, frame_y)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            show = False

                    pbar.set_postfix(
                        label=f"{label_name} ({conf_value:.2f})", time=f"{start_time_sec:.1f}-{end_time_sec:.1f}s")
                    pbar.update(1)
                    if fn_push is not None:
                        fn_push(["Processing temporal blocks",
                                pbar.n, total_blocks])
        # Print all predictions
        if debug:
            print("All predictions from temporal blocks:")
            for i, (label, conf, start_t, end_t, dur) in enumerate(all_predictions):
                print(
                    f"{i+1}: {label} ({conf:.2f}) [{start_t:.1f}-{end_t:.1f}s] ({dur}s block)")
        # Summarize best result using majority vote with average confidence from specified block duration
        if all_predictions:
            filtered_predictions = [
                pred for pred in all_predictions if pred[4] == block_duration_for_summary]
            if filtered_predictions:
                label_counts = Counter(
                    [p for p, c, _, _, _ in filtered_predictions])
                label_avg_conf = {}
                for p in label_counts:
                    confs = [c for pred, c, _, _,
                             _ in filtered_predictions if pred == p]
                    label_avg_conf[p] = sum(confs) / len(confs)
                seen = OrderedDict()
                for p, c, s, e, d in filtered_predictions:
                    if p not in seen:
                        # count and first start time
                        seen[p] = (label_counts[p], s)
                # Find the maximum count
                if seen:
                    max_count = max(v[0] for v in seen.values())
                    # Filter labels with count >= 2 (to keep significant ones)
                    candidate_labels = [(p, v)
                                        for p, v in seen.items() if v[0] >= 2]
                    # Sort by first appearance time ascending, then by average confidence descending
                    sorted_labels = sorted(candidate_labels, key=lambda x: (
                        x[1][1], -label_avg_conf[x[0]]))
                    unique_predictions = [(p, label_avg_conf[p])
                                          for p, _ in sorted_labels]
                else:
                    unique_predictions = []
                if debug:
                    print(
                        f"Best summary (majority vote with avg confidence from {block_duration_for_summary}s blocks): {' '.join([f'{p}({c:.2f})' for p, c in unique_predictions])}")
            else:
                unique_predictions = []
                if debug:
                    print(
                        f"No predictions from {block_duration_for_summary}s blocks.")
        else:
            unique_predictions = []

        if show:
            cv2.destroyAllWindows()

        # Gộp text (key) thành một chuỗi
        text = " ".join([k for k, _ in unique_predictions])

        # Tính trung bình value
        confidence = sum(v for _, v in unique_predictions) / \
            len(unique_predictions) if unique_predictions else float('inf')

        return text, confidence, unique_predictions

    @staticmethod
    def load_model(model_path='models/abc_vsl.pth', device=None, debug=False):
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if debug:
            print(f"Using device: {device}")

        # Initialize model
        model = ConvNeXtTransformer()
        ckpt = torch.load(model_path, map_location=device)

        ckpt = torch.load(model_path, map_location=device)

        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        from collections import OrderedDict
        if any(k.startswith("module.") for k in state.keys()):
            new_state = OrderedDict((k.replace("module.", "", 1), v)
                                    for k, v in state.items())
            state = new_state

        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0 and debug:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        model.to(device)
        model.eval()

        epoch = ckpt.get("epoch")
        idx_to_label = ckpt.get("idx_to_label")
        best_f1 = ckpt.get("best_f1")
        predict_steps = ckpt.get("predict_steps")
        f1 = ckpt.get("f1")
        recall = ckpt.get("recall")
        precision = ckpt.get("precision")
        if debug:
            print(f"Model F1: {f1}, Recall: {recall}, Precision: {precision}")
            print(f"Model loaded from epoch {epoch}, best F1: {best_f1}")
            print(f"Label mapping: {idx_to_label}")
            print(f"Predict steps: {predict_steps}")

        model.device = device
        model.epoch = epoch
        model.idx_to_label = idx_to_label
        model.best_f1 = best_f1
        model.predict_steps = predict_steps
        model.f1 = f1
        model.recall = recall
        model.precision = precision

        return model
