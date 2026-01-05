import os
import pickle
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
import mediapipe as mp
from torch.optim import AdamW
from torchvision import models
import torch.nn.functional as F
import mediapipe.tasks as mp_tasks
from torchvision import transforms as T
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

TARGET_FRAMES = 16


def convert_pickle_to_json(src_path='dataset/label_mapping.pkl', dest_path='dataset/label_mapping.json'):
    with open(src_path, 'rb') as f:
        data = pickle.load(f)

    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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
                    pbar.update(1)
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
        with open(f"{save_directory}{output_csv}", mode='w', newline='', encoding='utf-8') as f:
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
            f'{save_directory}checkpoints/checkpoint_epoch_*.pth')
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
            checkpoint_path = f'{save_directory}checkpoints/checkpoint_epoch_{resume_epoch}.pth'
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
        results_file = f'{save_directory}results.csv'
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
            os.makedirs(f'{save_directory}checkpoints', exist_ok=True)
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
            checkpoint_path = f'{save_directory}checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            self.cleanup_old_checkpoints(
                max_checkpoints, save_directory=save_directory)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                # Also save as best_model.pth
                torch.save(checkpoint, f"{save_directory}{save_path}")
                print(f"✓ Best model saved with F1: {best_f1:.2f}%")

        return model

    def summarize_best_result(self, all_predictions, block_duration_for_summary, debug=True):
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

        return unique_predictions

    def predict_sign_language_sentence(self, video_path, fn_push=None, window_size=16, stride=8, confidence_threshold=0.0, block_durations=None, target_fps=None, block_duration_for_summary=1, show=False, debug=False):
        model = self.to(self.device)

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
                                pbar.n, total_blocks, all_predictions])
        # Print all predictions
        if debug:
            print("All predictions from temporal blocks:")
            for i, (label, conf, start_t, end_t, dur) in enumerate(all_predictions):
                print(
                    f"{i+1}: {label} ({conf:.2f}) [{start_t:.1f}-{end_t:.1f}s] ({dur}s block)")
        # Summarize best result using majority vote with average confidence from specified block duration
        unique_predictions = self.summarize_best_result(
            all_predictions, block_duration_for_summary, debug=debug)

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


class KeypointDataset(Dataset):
    def __init__(self, root_dir, label_to_idx_path, keypoints_cache_dir=None,
                 target_frames=16, training=False, pose_name='default', hand_name='default'):
        self.root_dir = root_dir
        if keypoints_cache_dir is None:
            keypoints_cache_dir = root_dir + '-json'
        self.keypoints_cache_dir = keypoints_cache_dir
        self.target_frames = target_frames
        self.training = training
        self.pose_name = pose_name
        self.hand_name = hand_name

        self.instances, self.labels, self.label_idx = [], [], []

        with open(label_to_idx_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        self.label_mapping = {unicodedata.normalize(
            'NFC', k): v for k, v in self.label_mapping.items()}

        for label_folder in sorted(os.listdir(root_dir))[:len(self.label_mapping)]:
            path = os.path.join(root_dir, label_folder)
            if os.path.isdir(path):
                for video_file in os.listdir(path):
                    video_path = os.path.join(path, video_file)
                    self.instances.append(video_path)
                    self.labels.append(label_folder)
                    self.label_idx.append(
                        self.label_mapping[unicodedata.normalize('NFC', label_folder)])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        video_path = self.instances[idx]
        # Create relative path for cache
        relative_path = os.path.relpath(video_path, self.root_dir)
        base_name = relative_path.replace('.mp4', '').replace(
            '.avi', '').replace('.mov', '').replace('.mkv', '')
        cache_file = os.path.join(
            self.keypoints_cache_dir, f"{base_name}_{self.pose_name}_pose_{self.hand_name}_hand.json")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}. Run preprocess_keypoints first.")

        # Load from cache
        with open(cache_file, 'r') as f:
            frames_keypoints = json.load(f)

        # Convert to tensor
        keypoints = torch.tensor(
            frames_keypoints, dtype=torch.float32)  # (T, D)

        return {
            'keypoints': keypoints,
            'label_idx': self.label_idx[idx],
            'label': self.labels[idx]
        }


class KeypointTransformer(nn.Module):
    """
    Transformer for keypoints sequences
    Input: (B, T, D) = (B, 16, 225)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes=100, d_model=64, hidden_size=256):
        super().__init__()

        # Project keypoints to d_model
        self.input_proj = nn.Linear(225, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=64, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=4)  # More layers for keypoints

        self.attention_pool = AttentionPooling(d_model)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_classes)
        )

        self._init_weights()

        # Pose models: default, lite, heavy, full
        self.POSE_MODELS = {
            'default': None,  # Use mp.solutions.pose
            # 'lite': 'models/pose_landmarker_lite.task',
            # 'heavy': 'models/pose_landmarker_heavy.task',
            # 'full': 'models/pose_landmarker_full.task'
        }

        # Hand models: default, hand_landmarker
        self.HAND_MODELS = {
            'default': None,  # Use mp.solutions.hands
            # 'hand_landmarker': 'models/hand_landmarker.task'
        }

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
        B, T, D = x.shape

        x = self.input_proj(x)  # (B, T, d_model)

        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.attention_pool(x)

        x = self.fc(x)

        return x

    def init_pose_model(self, model_name):
        if model_name == 'default':
            mp_pose = mp.solutions.pose
            return mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        else:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=self.POSE_MODELS[model_name])
            options = mp_tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks.vision.RunningMode.IMAGE  # Use IMAGE mode
            )
            return mp_tasks.vision.PoseLandmarker.create_from_options(options)

    def init_hand_model(self, model_name):
        if model_name == 'default':
            mp_hands = mp.solutions.hands
            return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        else:
            try:
                base_options = mp_tasks.BaseOptions(
                    model_asset_path=self.HAND_MODELS[model_name])
                options = mp_tasks.vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_tasks.vision.RunningMode.IMAGE,
                    num_hands=2
                )
                return mp_tasks.vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(
                    f"Failed to load hand model {model_name}: {e}. Falling back to default.")
                return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    def extract_keypoints_from_frame(self, frame, pose_model, hand_model, pose_name, hand_name):
        """Extract keypoints using specified models

        Keypoint structure (225 values):
        - Pose landmarks (indices 0-98): 33 landmarks × 3 (x,y,z)
        0-2: nose, 3-5: left_eye_inner, 6-8: left_eye, 9-11: left_eye_outer,
        12-14: right_eye_inner, 15-17: right_eye, 18-20: right_eye_outer,
        21-23: left_ear, 24-26: right_ear, 27-29: mouth_left, 30-32: mouth_right,
        33-35: left_shoulder, 36-38: right_shoulder, 39-41: left_elbow, 42-44: right_elbow,
        45-47: left_wrist, 48-50: right_wrist, 51-53: left_pinky, 54-56: right_pinky,
        57-59: left_index, 60-62: right_index, 63-65: left_thumb, 66-68: right_thumb,
        69-71: left_hip, 72-74: right_hip, 75-77: left_knee, 78-80: right_knee,
        81-83: left_ankle, 84-86: right_ankle, 87-89: left_heel, 90-92: right_heel,
        93-95: left_foot_index, 96-98: right_foot_index
        - Hand 1 (indices 99-161): 21 landmarks × 3
        99-101: wrist, 102-104: thumb_cmc, 105-107: thumb_mcp, 108-110: thumb_ip, 111-113: thumb_tip,
        114-116: index_finger_mcp, 117-119: index_finger_pip, 120-122: index_finger_dip, 123-125: index_finger_tip,
        126-128: middle_finger_mcp, 129-131: middle_finger_pip, 132-134: middle_finger_dip, 135-137: middle_finger_tip,
        138-140: ring_finger_mcp, 141-143: ring_finger_pip, 144-146: ring_finger_dip, 147-149: ring_finger_tip,
        150-152: pinky_mcp, 153-155: pinky_pip, 156-158: pinky_dip, 159-161: pinky_tip
        - Hand 2 (indices 162-224): same as Hand 1
        """
        # Initialize with fixed size: pose (33*3=99) + hands (2*21*3=126) = 225
        keypoints = [0.0] * 225
        pose_results = None
        hands_results = None

        # Extract pose keypoints (indices 0-98)
        if pose_name == 'default':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose_model.process(frame_rgb)
            if pose_results.pose_landmarks:
                for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                    keypoints[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_results = pose_model.detect(
                mp_image)  # Use detect for IMAGE mode
            if pose_results.pose_landmarks:
                for i, lm in enumerate(pose_results.pose_landmarks[0]):
                    keypoints[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

        # Extract hand keypoints (indices 99-224: hand1 99-161, hand2 162-224)
        hand_start_idx = 99
        if hand_name == 'default':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_results = hand_model.process(frame_rgb)
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
                    for i, lm in enumerate(hand_landmarks.landmark):
                        start = hand_start_idx + hand_idx * 63 + i * 3
                        keypoints[start:start+3] = [lm.x, lm.y, lm.z]
        else:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hands_results = hand_model.detect(mp_image)
                if hands_results.hand_landmarks:
                    for hand_idx, hand in enumerate(hands_results.hand_landmarks[:2]):
                        for i, lm in enumerate(hand):
                            start = hand_start_idx + hand_idx * 63 + i * 3
                            keypoints[start:start+3] = [lm.x, lm.y, lm.z]
            except Exception as e:
                print(
                    f"Error detecting with hand model {hand_name}: {e}. Skipping hand keypoints.")
                # Keypoints remain as zeros

        return keypoints, pose_results, hands_results

    def keypoints_to_dict(self, keypoints):
        """Convert keypoints list (225 values) to dictionary with meaningful names

        Args:
            keypoints: list of 225 float values

        Returns:
            dict: key is landmark name with coordinate, value is the coordinate value
        """
        if len(keypoints) != 225:
            raise ValueError(f"Expected 225 keypoints, got {len(keypoints)}")

        # Define landmark names as variables
        pose_landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]

        hand_landmark_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
            'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
            'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]

        result = {}

        # Pose keypoints (0-98)
        for i, name in enumerate(pose_landmark_names):
            base_idx = i * 3
            result[f'pose_{name}_x'] = keypoints[base_idx]
            result[f'pose_{name}_y'] = keypoints[base_idx + 1]
            result[f'pose_{name}_z'] = keypoints[base_idx + 2]

        # Hand keypoints (99-224)
        for hand_idx in range(2):
            hand_prefix = f'hand{hand_idx + 1}'
            for i, name in enumerate(hand_landmark_names):
                base_idx = 99 + hand_idx * 63 + i * 3
                result[f'{hand_prefix}_{name}_x'] = keypoints[base_idx]
                result[f'{hand_prefix}_{name}_y'] = keypoints[base_idx + 1]
                result[f'{hand_prefix}_{name}_z'] = keypoints[base_idx + 2]

        return result

    def read_video_and_extract_keypoints(self, video_path, pose_model, hand_model, pose_name, hand_name, target_frames=16, show=False):
        """Read video and extract keypoints for all frames"""
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0:
            fps = 30  # Default fallback
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            keypoints, pose_results, hands_results = self.extract_keypoints_from_frame(
                frame, pose_model, hand_model, pose_name, hand_name)
            frames_keypoints.append(keypoints)
            frame_count += 1

            if show:
                mp_pose = mp.solutions.pose
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                frame_copy = frame.copy()
                # Draw pose
                if pose_name == 'default' and pose_results and pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                elif pose_name != 'default' and pose_results and pose_results.pose_landmarks:
                    # For Tasks API, pose_landmarks is list of NormalizedLandmark
                    # Need to convert to landmark format for drawing
                    # This might be tricky, perhaps skip drawing for Tasks API or implement conversion
                    pass  # Skip drawing for Tasks API for now

                # Draw hands
                if hand_name == 'default' and hands_results and hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                elif hand_name != 'default' and hands_results and hands_results.hand_landmarks:
                    # Similarly for Tasks API hands
                    pass  # Skip drawing for Tasks API for now

                cv2.imshow('MediaPipe Keypoints', frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

        if len(frames_keypoints) == 0:
            raise ValueError(f"Could not extract keypoints from {video_path}")

        # Downsample to target_frames
        total = len(frames_keypoints)
        if total >= target_frames:
            indices = torch.linspace(0, total - 1, target_frames).long()
        else:
            indices = torch.arange(total)
            pad = target_frames - total
            indices = torch.cat([indices, indices[-1].repeat(pad)])

        frames_keypoints = [frames_keypoints[i] for i in indices.tolist()]

        return frames_keypoints, fps

    def preprocess_keypoints(self, root_dir, label_to_idx_path, keypoints_cache_dir=None, show=False, force_recreate=False, multiple_mp=False):
        """Preprocess all videos to extract keypoints and save to JSON cache"""
        if keypoints_cache_dir is None:
            keypoints_cache_dir = root_dir + '-json'
        os.makedirs(keypoints_cache_dir, exist_ok=True)

        with open(label_to_idx_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        label_mapping = {unicodedata.normalize(
            'NFC', k): v for k, v in label_mapping.items()}

        total_videos = 0

        if multiple_mp:
            pose_keys = self.POSE_MODELS.keys()
            hand_keys = self.HAND_MODELS.keys()
        else:
            pose_keys = ['default']
            hand_keys = ['default']

        for pose_name in pose_keys:
            pose_model = self.init_pose_model(pose_name)
            for hand_name in hand_keys:
                hand_model = self.init_hand_model(hand_name)
                for item in sorted(os.listdir(root_dir)):
                    path = os.path.join(root_dir, item)
                    if os.path.isdir(path):
                        label_folder = item
                        for video_file in os.listdir(path):
                            video_path = os.path.join(path, video_file)
                            # Create relative path for cache
                            relative_path = os.path.relpath(
                                video_path, root_dir)
                            base_name = relative_path.replace('.mp4', '').replace(
                                '.avi', '').replace('.mov', '').replace('.mkv', '')

                            cache_file = os.path.join(
                                keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")
                            os.makedirs(os.path.dirname(
                                cache_file), exist_ok=True)

                            if force_recreate or not os.path.exists(cache_file):
                                try:
                                    frames_keypoints = self.read_video_and_extract_keypoints(
                                        video_path, pose_model, hand_model, pose_name, hand_name, TARGET_FRAMES, show=show
                                    )
                                    with open(cache_file, 'w') as f:
                                        json.dump(frames_keypoints, f)
                                    total_videos += 1
                                    print(
                                        f"Processed {total_videos}: {video_file} with {pose_name} pose and {hand_name} hand")
                                except Exception as e:
                                    print(
                                        f"Error processing {video_file} with {pose_name}/{hand_name}: {e}")
                            else:
                                print(
                                    f"Cache exists for {video_file} with {pose_name}/{hand_name}, skipping")
                    elif os.path.isfile(path) and item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_file = item
                        video_path = path
                        relative_path = item
                        base_name = item.replace('.mp4', '').replace(
                            '.avi', '').replace('.mov', '').replace('.mkv', '')
                        cache_file = os.path.join(
                            keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

                        if force_recreate or not os.path.exists(cache_file):
                            try:
                                frames_keypoints = self.read_video_and_extract_keypoints(
                                    video_path, pose_model, hand_model, pose_name, hand_name, TARGET_FRAMES, show=show
                                )
                                with open(cache_file, 'w') as f:
                                    json.dump(frames_keypoints, f)
                                total_videos += 1
                                print(
                                    f"Processed {total_videos}: {video_file} with {pose_name} pose and {hand_name} hand")
                            except Exception as e:
                                print(
                                    f"Error processing {video_file} with {pose_name}/{hand_name}: {e}")
                        else:
                            print(
                                f"Cache exists for {video_file} with {pose_name}/{hand_name}, skipping")

        print(
            f"Preprocessing complete. Total new videos processed: {total_videos}")

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

    def collate_fn_keypoints(self, batch):
        """Custom collate function for keypoints batch"""
        keypoints = torch.stack([item['keypoints'] for item in batch])
        labels = torch.tensor([item['label_idx'] for item in batch])
        label_names = [item['label'] for item in batch]
        return {'keypoints': keypoints, 'label_idx': labels, 'label': label_names}

    def cleanup_old_checkpoints(self, max_checkpoints, save_directory=""):
        """Remove old checkpoints, keeping only the latest max_checkpoints"""
        if max_checkpoints is None or max_checkpoints <= 0:
            return

        os.makedirs('mp_checkpoints', exist_ok=True)
        checkpoint_files = glob.glob(
            f'{save_directory}mp_checkpoints/mp_checkpoint_epoch_*.pth')
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

    def evaluate_keypoints(self, folder_path, label_to_idx_path, output_csv="predictions.csv",
                           device='cuda', model_path=None, target_frames=16, keypoints_cache_dir=None, show=False, pose_name='default', hand_name='default'):
        model = self.to(device)

        """Evaluate trained model on test set using keypoints"""
        if keypoints_cache_dir is None:
            keypoints_cache_dir = folder_path + '-json'
        if model_path:
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")

        model.eval()

        with open(label_to_idx_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        idx_to_label = {v: k for k, v in label_mapping.items()}

        video_files = sorted([f for f in os.listdir(
            folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        print(f"Found {len(video_files)} videos in '{folder_path}'")

        predictions = []

        with torch.no_grad():
            for video_file in tqdm(video_files, desc="Predicting"):
                video_path = os.path.join(folder_path, video_file)
                # Create relative path for cache (assuming flat structure for test, but to be safe)
                relative_path = os.path.relpath(video_path, folder_path)
                base_name = relative_path.replace('.mp4', '').replace(
                    '.avi', '').replace('.mov', '').replace('.mkv', '')
                cache_file = os.path.join(
                    keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")

                if not os.path.exists(cache_file):
                    raise FileNotFoundError(
                        f"Cache file not found: {cache_file}. Run preprocess_keypoints first.")

                with open(cache_file, 'r') as f:
                    frames_keypoints = json.load(f)

                keypoints = torch.tensor(
                    frames_keypoints, dtype=torch.float32).unsqueeze(0).to(device)

                outputs = model(keypoints)
                _, predicted = outputs.max(1)
                label_idx = predicted.item()
                label_name = idx_to_label[label_idx]

                predictions.append((video_file, label_name))

        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_name', 'label'])
            writer.writerows(predictions)

        print(f"\nPredictions saved to '{output_csv}'")
        print(f"Total videos processed: {len(predictions)}")

    def train_epoch_keypoints(self, dataloader, criterion, optimizer, device='cuda'):
        """One training epoch for keypoints"""
        model = self.to(device)
        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc='Training')
        for batch in progress:
            keypoints, labels = batch['keypoints'].to(
                device), batch['label_idx'].to(device)
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(
                {'loss': f'{total_loss / (len(progress)+1e-9):.4f}'})
        return total_loss / len(dataloader)

    def validate_keypoints(self, dataloader, criterion, device='cuda'):
        """Validation for keypoints"""
        model = self.to(device)
        model.eval()
        total_loss, preds, labels_all = 0, [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                keypoints, labels = batch['keypoints'].to(
                    device), batch['label_idx'].to(device)
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_all, preds, average='macro', zero_division=0)
        return total_loss / len(dataloader), {'precision': precision*100, 'recall': recall*100, 'f1': f1*100}

    def train_keypoint_model(self, train_loader, val_loader,
                             num_epochs=20, lr=1e-4, device='cuda', save_path='best_mp_model.pth', label_mapping_path='dataset/label_mapping.json', resume_epoch=None, max_checkpoints=None, save_directory=""):
        """Full training loop for keypoints model"""
        model = self.to(device)

        label_mapping_file = label_mapping_path
        label_mapping = {}
        if label_mapping_file.endswith('.json'):
            with open(label_mapping_file, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)

        # Resume from checkpoint if provided
        start_epoch = 0
        best_f1 = 0.0
        if resume_epoch is not None:
            checkpoint_path = f'{save_directory}mp_checkpoints/mp_checkpoint_epoch_{resume_epoch}.pth'
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
        results_file = 'results.csv'
        file_exists = os.path.exists(results_file)
        with open(results_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss',
                                'precision', 'recall', 'f1'])

        for epoch in range(start_epoch, num_epochs):
            print(f"\n===== Epoch {epoch+1}/{num_epochs} ======")
            train_loss = self.train_epoch_keypoints(
                train_loader, criterion, optimizer, device)
            val_loss, val_metrics = self.validate_keypoints(
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
            os.makedirs(f'{save_directory}mp_checkpoints', exist_ok=True)
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
            checkpoint_path = f'{save_directory}mp_checkpoints/mp_checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            self.cleanup_old_checkpoints(max_checkpoints)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                # Also save as best_mp_model.pth
                torch.save(checkpoint, f"{save_directory}{save_path}")
                print(f"✓ Best model saved with F1: {best_f1:.2f}%")

        return model

    def downsample_keypoints(self, frames_keypoints, target_frames):
        """Downsample keypoints to target number of frames"""
        total = len(frames_keypoints)
        if total >= target_frames:
            indices = torch.linspace(0, total - 1, target_frames).long()
        else:
            indices = torch.arange(total)
            pad = target_frames - total
            indices = torch.cat([indices, indices[-1].repeat(pad)])

        return [frames_keypoints[i] for i in indices.tolist()]

    def read_video_and_extract_all_keypoints(self, video_path, pose_model, hand_model, pose_name, hand_name, target_fps=16, show=False):
        """Read video and extract keypoints for all frames"""
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback

        # Calculate sampling interval for reframing
        if target_fps < fps:
            sample_interval = fps / target_fps
            next_sample = 0.0
        else:
            sample_interval = 1.0
            next_sample = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count >= next_sample:
                keypoints, pose_results, hands_results = self.extract_keypoints_from_frame(
                    frame, pose_model, hand_model, pose_name, hand_name)
                frames_keypoints.append(keypoints)
                next_sample += sample_interval

                if show:
                    mp_pose = mp.solutions.pose
                    mp_hands = mp.solutions.hands
                    mp_drawing = mp.solutions.drawing_utils
                    frame_copy = frame.copy()
                    # Draw pose
                    if pose_name == 'default' and pose_results and pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    elif pose_name != 'default' and pose_results and pose_results.pose_landmarks:
                        pass  # Skip drawing for Tasks API for now

                    # Draw hands
                    if hand_name == 'default' and hands_results and hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    elif hand_name != 'default' and hands_results and hands_results.hand_landmarks:
                        pass  # Skip drawing for Tasks API for now

                    cv2.imshow('MediaPipe Keypoints', frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_count += 1

        cap.release()

        if len(frames_keypoints) == 0:
            raise ValueError(f"Could not extract keypoints from {video_path}")

        return frames_keypoints, fps

    def predict_sign_language(self, video_path, device='cuda', show=False, block_durations=None, confidence_threshold=0.0, target_fps=16, block_duration_for_summary=1, debug=False):
        """Predict sign language from video using keypoints with temporal block analysis

        Args:
            video_path: Path to input video
            model_path: Path to trained model
            label_mapping_path: Path to label mapping
            device: Device to run on (cuda/cpu)
            show: Show visualization during processing
            block_durations: List of block durations in seconds (default [1, 2, 3])
            confidence_threshold: Minimum confidence to include prediction
            block_duration_for_summary: Which block duration to use for final summary (1, 2, or 3)

        Returns:
            List of (label_name, confidence) tuples
        """

        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        pose_model = mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5)
        hands_model = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        # Extract keypoints with FPS
        print(f"Extracting keypoints from {video_path}...")
        frames_keypoints_all, fps = self.read_video_and_extract_all_keypoints(
            video_path, pose_model, hands_model, 'default', 'default', target_fps=target_fps, show=show)

        total_frames = len(frames_keypoints_all)
        print(f"Total frames: {total_frames}, FPS: {fps}")
        fps = target_fps

        # Default block durations
        if block_durations is None:
            block_durations = [1, 2, 3]

        # Temporal block analysis with sliding windows
        shift_duration = 0.5  # seconds
        shift_frames = max(int(fps * shift_duration), 1)

        all_predictions = []
        total_blocks = sum((total_frames - int(fps * d) + 1) //
                           shift_frames + 1 for d in block_durations if int(fps * d) > 0)
        print(f"Processing {total_blocks} temporal blocks...")
        with tqdm(total=total_blocks, desc="Temporal block analysis") as pbar:
            for duration in block_durations:
                frames_per_block = int(fps * duration)
                if frames_per_block == 0:
                    frames_per_block = fps * duration  # Fallback

                for start in range(0, total_frames - frames_per_block + 1, shift_frames):
                    end = min(start + frames_per_block, total_frames)
                    start_time_sec = start / fps
                    end_time_sec = end / fps

                    # Extract keypoints for this block
                    block_keypoints = frames_keypoints_all[start:end]

                    # Downsample to TARGET_FRAMES if needed
                    if len(block_keypoints) != target_fps:
                        block_keypoints = self.downsample_keypoints(
                            block_keypoints, target_fps)

                    # Convert to tensor and predict
                    keypoints_tensor = torch.tensor(
                        # (1, T, D)
                        block_keypoints, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = self(keypoints_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        confidence, predicted = probs.max(1)
                        label_idx = predicted.item()
                        label_name = self.idx_to_label[label_idx]
                        conf_value = confidence.item()

                        if conf_value >= confidence_threshold:
                            all_predictions.append(
                                (label_name, conf_value, start_time_sec, end_time_sec, duration))

                    pbar.set_postfix(
                        label=f"{label_name} ({conf_value:.2f})", time=f"{start_time_sec:.1f}-{end_time_sec:.1f}s")
                    pbar.update(1)

        # Print all predictions
        print("\nAll predictions from temporal blocks:")
        for i, (label, conf, start_t, end_t, dur) in enumerate(all_predictions):
            print(
                f"{i+1}: {label} ({conf:.2f}) [{start_t:.1f}-{end_t:.1f}s] ({dur}s block)")

        # Summarize using majority vote
        unique_predictions = []
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
                        seen[p] = (label_counts[p], s)

                if seen:
                    candidate_labels = [(p, v)
                                        for p, v in seen.items() if v[0] >= 2]
                    if candidate_labels:
                        sorted_labels = sorted(candidate_labels, key=lambda x: (
                            x[1][1], -label_avg_conf[x[0]]))
                        unique_predictions = [(p, label_avg_conf[p])
                                              for p, _ in sorted_labels]

                print(
                    f"\nBest summary (majority vote from {block_duration_for_summary}s blocks): {' '.join([f'{p}({c:.2f})' for p, c in unique_predictions])}")
            else:
                print(
                    f"No predictions from {block_duration_for_summary}s blocks.")

        if show:
            cv2.destroyAllWindows()

        return unique_predictions

    @staticmethod
    def load_model(model_path='models/abc_vsl.pth', device=None, debug=False):
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if debug:
            print(f"Using device: {device}")

        # Initialize model
        model = KeypointTransformer()
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
