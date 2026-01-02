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
import mediapipe as mp
import argparse
import warnings
warnings.filterwarnings('ignore')

NUM_CLASSES = 100
TARGET_FRAMES = 16

import mediapipe as mp
import mediapipe.tasks as mp_tasks
import warnings
warnings.filterwarnings('ignore')

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Pose models: default, lite, heavy, full
POSE_MODELS = {
    'default': None,  # Use mp.solutions.pose
    # 'lite': 'models/pose_landmarker_lite.task',
    # 'heavy': 'models/pose_landmarker_heavy.task',
    # 'full': 'models/pose_landmarker_full.task'
}

# Hand models: default, hand_landmarker
HAND_MODELS = {
    'default': None,  # Use mp.solutions.hands
    # 'hand_landmarker': 'models/hand_landmarker.task'
}
mp_drawing = mp.solutions.drawing_utils

def init_pose_model(model_name):
    if model_name == 'default':
        return mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    else:
        base_options = mp_tasks.BaseOptions(model_asset_path=POSE_MODELS[model_name])
        options = mp_tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks.vision.RunningMode.IMAGE  # Use IMAGE mode
        )
        return mp_tasks.vision.PoseLandmarker.create_from_options(options)

def init_hand_model(model_name):
    if model_name == 'default':
        return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    else:
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODELS[model_name])
            options = mp_tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks.vision.RunningMode.IMAGE,
                num_hands=2
            )
            return mp_tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to load hand model {model_name}: {e}. Falling back to default.")
            return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_keypoints_from_frame(frame, pose_model, hand_model, pose_name, hand_name):
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_results = pose_model.detect(mp_image)  # Use detect for IMAGE mode
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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands_results = hand_model.detect(mp_image)
            if hands_results.hand_landmarks:
                for hand_idx, hand in enumerate(hands_results.hand_landmarks[:2]):
                    for i, lm in enumerate(hand):
                        start = hand_start_idx + hand_idx * 63 + i * 3
                        keypoints[start:start+3] = [lm.x, lm.y, lm.z]
        except Exception as e:
            print(f"Error detecting with hand model {hand_name}: {e}. Skipping hand keypoints.")
            # Keypoints remain as zeros

    return keypoints, pose_results, hands_results

def keypoints_to_dict(keypoints):
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

def cleanup_old_checkpoints(max_checkpoints):
    """Remove old checkpoints, keeping only the latest max_checkpoints"""
    if max_checkpoints is None or max_checkpoints <= 0:
        return
    
    os.makedirs('mp_checkpoints', exist_ok=True)
    checkpoint_files = glob.glob('mp_checkpoints/mp_checkpoint_epoch_*.pth')
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

def read_video_and_extract_keypoints(video_path, pose_model, hand_model, pose_name, hand_name, target_frames=16, show=False):
    """Read video and extract keypoints for all frames"""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints, pose_results, hands_results = extract_keypoints_from_frame(frame, pose_model, hand_model, pose_name, hand_name)
        frames_keypoints.append(keypoints)
        frame_count += 1

        if show:
            frame_copy = frame.copy()
            # Draw pose
            if pose_name == 'default' and pose_results and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            elif pose_name != 'default' and pose_results and pose_results.pose_landmarks:
                # For Tasks API, pose_landmarks is list of NormalizedLandmark
                # Need to convert to landmark format for drawing
                # This might be tricky, perhaps skip drawing for Tasks API or implement conversion
                pass  # Skip drawing for Tasks API for now

            # Draw hands
            if hand_name == 'default' and hands_results and hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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

    return frames_keypoints

def preprocess_keypoints(root_dir, label_to_idx_path, keypoints_cache_dir=None, show=False, force_recreate=False):
    """Preprocess all videos to extract keypoints and save to JSON cache"""
    if keypoints_cache_dir is None:
        keypoints_cache_dir = root_dir + '-json'
    os.makedirs(keypoints_cache_dir, exist_ok=True)

    with open(label_to_idx_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    label_mapping = {unicodedata.normalize('NFC', k): v for k, v in label_mapping.items()}

    total_videos = 0
    for pose_name in POSE_MODELS.keys():
        pose_model = init_pose_model(pose_name)
        for hand_name in HAND_MODELS.keys():
            hand_model = init_hand_model(hand_name)
            for item in sorted(os.listdir(root_dir)):
                path = os.path.join(root_dir, item)
                if os.path.isdir(path):
                    label_folder = item
                    for video_file in os.listdir(path):
                        video_path = os.path.join(path, video_file)
                        # Create relative path for cache
                        relative_path = os.path.relpath(video_path, root_dir)
                        base_name = relative_path.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')

                        cache_file = os.path.join(keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

                        if force_recreate or not os.path.exists(cache_file):
                            try:
                                frames_keypoints = read_video_and_extract_keypoints(
                                    video_path, pose_model, hand_model, pose_name, hand_name, TARGET_FRAMES, show=show
                                )
                                with open(cache_file, 'w') as f:
                                    json.dump(frames_keypoints, f)
                                total_videos += 1
                                print(f"Processed {total_videos}: {video_file} with {pose_name} pose and {hand_name} hand")
                            except Exception as e:
                                print(f"Error processing {video_file} with {pose_name}/{hand_name}: {e}")
                        else:
                            print(f"Cache exists for {video_file} with {pose_name}/{hand_name}, skipping")
                elif os.path.isfile(path) and item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_file = item
                    video_path = path
                    relative_path = item
                    base_name = item.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')
                    cache_file = os.path.join(keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

                    if force_recreate or not os.path.exists(cache_file):
                        try:
                            frames_keypoints = read_video_and_extract_keypoints(
                                video_path, pose_model, hand_model, pose_name, hand_name, TARGET_FRAMES, show=show
                            )
                            with open(cache_file, 'w') as f:
                                json.dump(frames_keypoints, f)
                            total_videos += 1
                            print(f"Processed {total_videos}: {video_file} with {pose_name} pose and {hand_name} hand")
                        except Exception as e:
                            print(f"Error processing {video_file} with {pose_name}/{hand_name}: {e}")
                    else:
                        print(f"Cache exists for {video_file} with {pose_name}/{hand_name}, skipping")

    print(f"Preprocessing complete. Total new videos processed: {total_videos}")

def collate_fn_keypoints(batch):
    """Custom collate function for keypoints batch"""
    keypoints = torch.stack([item['keypoints'] for item in batch])
    labels = torch.tensor([item['label_idx'] for item in batch])
    label_names = [item['label'] for item in batch]
    return {'keypoints': keypoints, 'label_idx': labels, 'label': label_names}

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
        self.label_mapping = {unicodedata.normalize('NFC', k): v for k, v in self.label_mapping.items()}

        for label_folder in sorted(os.listdir(root_dir))[:NUM_CLASSES]:
            path = os.path.join(root_dir, label_folder)
            if os.path.isdir(path):
                for video_file in os.listdir(path):
                    video_path = os.path.join(path, video_file)
                    self.instances.append(video_path)
                    self.labels.append(label_folder)
                    self.label_idx.append(self.label_mapping[unicodedata.normalize('NFC', label_folder)])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        video_path = self.instances[idx]
        # Create relative path for cache
        relative_path = os.path.relpath(video_path, self.root_dir)
        base_name = relative_path.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')
        cache_file = os.path.join(self.keypoints_cache_dir, f"{base_name}_{self.pose_name}_pose_{self.hand_name}_hand.json")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}. Run preprocess_keypoints first.")

        # Load from cache
        with open(cache_file, 'r') as f:
            frames_keypoints = json.load(f)

        # Convert to tensor
        keypoints = torch.tensor(frames_keypoints, dtype=torch.float32)  # (T, D)

        return {
            'keypoints': keypoints,
            'label_idx': self.label_idx[idx],
            'label': self.labels[idx]
        }

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
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    """Attention pooling"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

class KeypointTransformer(nn.Module):
    """
    Transformer for keypoints sequences
    Input: (B, T, D) = (B, 16, 225)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=100, d_model=64, hidden_size=256):
        super().__init__()

        self.input_proj = nn.Linear(225, d_model)  # Project keypoints to d_model

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=64, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # More layers for keypoints

        self.attention_pool = AttentionPooling(d_model)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_classes)
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
        B, T, D = x.shape

        x = self.input_proj(x)  # (B, T, d_model)

        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.attention_pool(x)

        x = self.fc(x)

        return x

def evaluate_keypoints(model, folder_path, label_to_idx_path, output_csv="predictions.csv",
                       device='cuda', model_path=None, target_frames=16, keypoints_cache_dir=None, show=False, pose_name='default', hand_name='default'):
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

    model = model.to(device)
    model.eval()

    with open(label_to_idx_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    video_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    print(f"Found {len(video_files)} videos in '{folder_path}'")

    predictions = []

    with torch.no_grad():
        for video_file in tqdm(video_files, desc="Predicting"):
            video_path = os.path.join(folder_path, video_file)
            # Create relative path for cache (assuming flat structure for test, but to be safe)
            relative_path = os.path.relpath(video_path, folder_path)
            base_name = relative_path.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')
            cache_file = os.path.join(keypoints_cache_dir, f"{base_name}_{pose_name}_pose_{hand_name}_hand.json")

            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"Cache file not found: {cache_file}. Run preprocess_keypoints first.")

            with open(cache_file, 'r') as f:
                frames_keypoints = json.load(f)

            keypoints = torch.tensor(frames_keypoints, dtype=torch.float32).unsqueeze(0).to(device)

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

def train_epoch_keypoints(model, dataloader, criterion, optimizer, device='cuda'):
    """One training epoch for keypoints"""
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc='Training')
    for batch in progress:
        keypoints, labels = batch['keypoints'].to(device), batch['label_idx'].to(device)
        optimizer.zero_grad()
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{total_loss / (len(progress)+1e-9):.4f}'})
    return total_loss / len(dataloader)

def validate_keypoints(model, dataloader, criterion, device='cuda'):
    """Validation for keypoints"""
    model.eval()
    total_loss, preds, labels_all = 0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            keypoints, labels = batch['keypoints'].to(device), batch['label_idx'].to(device)
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average='macro', zero_division=0)
    return total_loss / len(dataloader), {'precision': precision*100, 'recall': recall*100, 'f1': f1*100}

def train_keypoint_model(model, train_loader, val_loader,
                         num_epochs=20, lr=1e-4, device='cuda', save_path='best_mp_model.pth', resume_epoch=None, max_checkpoints=None):
    """Full training loop for keypoints model"""
    model = model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_f1 = 0.0
    if resume_epoch is not None:
        checkpoint_path = f'mp_checkpoints/mp_checkpoint_epoch_{resume_epoch}.pth'
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
        train_loss = train_epoch_keypoints(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_keypoints(model, val_loader, criterion, device)
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
        os.makedirs('mp_checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1
        }
        checkpoint_path = f'mp_checkpoints/mp_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(max_checkpoints)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # Also save as best_mp_model.pth
            torch.save(checkpoint, save_path)
            print(f"✓ Best model saved with F1: {best_f1:.2f}%")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MediaPipe Keypoint Model')
    parser.add_argument('--show', action='store_true', help='Show MediaPipe processing visualization')
    parser.add_argument('--force-recreate', action='store_true', help='Force recreate cache files even if they exist')
    parser.add_argument('--resume', type=int, default=None, help='Epoch number to resume training from (loads mp_checkpoints/mp_checkpoint_epoch_{epoch}.pth)')
    parser.add_argument('--max-checkpoints', type=int, default=5, help='Maximum number of checkpoints to keep (default: keep all)')
    args = parser.parse_args()

    # Preprocess keypoints for train dataset
    print("Preprocessing keypoints for train dataset...")
    preprocess_keypoints('dataset/train', 'dataset/label_mapping.json', show=args.show, force_recreate=args.force_recreate)

    # Preprocess keypoints for public test
    print("Preprocessing keypoints for public test...")
    preprocess_keypoints('dataset/public_test', 'dataset/label_mapping.json', show=args.show, force_recreate=args.force_recreate)

    # Preprocess keypoints for private test
    print("Preprocessing keypoints for private test...")
    preprocess_keypoints('dataset/private_test', 'dataset/label_mapping.json', show=args.show, force_recreate=args.force_recreate)

    # Tạo datasets với keypoints
    train_dataset_base = KeypointDataset(
        'dataset/train',
        'dataset/label_mapping.json',
        target_frames=TARGET_FRAMES,
        training=True,
        pose_name='default',
        hand_name='default'
    )

    val_dataset_base = KeypointDataset(
        'dataset/train',
        'dataset/label_mapping.json',
        target_frames=TARGET_FRAMES,
        training=False,
        pose_name='default',
        hand_name='default'
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
        batch_size=32,  # Larger batch for keypoints
        sampler=balanced_sampler,
        collate_fn=collate_fn_keypoints,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_keypoints,
        num_workers=4
    )

    print(f"Train: {len(train_dataset)} (balanced sampling)")
    print(f"Val: {len(val_dataset)}")

    model = KeypointTransformer(num_classes=NUM_CLASSES, d_model=64, hidden_size=256)

    model = train_keypoint_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        lr=1e-4,
        device='cuda',
        save_path='best_mp_model.pth',
        resume_epoch=args.resume,
        max_checkpoints=args.max_checkpoints
    )

    # Export public result
    evaluate_keypoints(
        model=model,
        folder_path="dataset/public_test",
        label_to_idx_path="dataset/label_mapping.json",
        model_path="best_mp_model.pth",
        output_csv="public_test_mp.csv",
        device="cuda",
        target_frames=16,
        show=args.show,
        pose_name='default',
        hand_name='default'
    )

    # Export private result
    evaluate_keypoints(
        model=model,
        folder_path="dataset/private_test",
        label_to_idx_path="dataset/label_mapping.json",
        model_path="best_mp_model.pth",
        output_csv="private_test_mp.csv",
        device="cuda",
        target_frames=16,
        show=args.show,
        pose_name='default',
        hand_name='default'
    )

    if args.show:
        cv2.destroyAllWindows()