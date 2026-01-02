# Import required libraries
import os
from time import time
import pickle
import json
from collections import Counter, OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models
import argparse
from tqdm import tqdm
from torchvision import transforms as T
import traceback
import ctypes
import math
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

# Constants
NUM_CLASSES = 100
TARGET_FRAMES = 16  # number of frames per video

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_frame(frame, pose_model, hands_model, show=False):
    """Extract arm and hand keypoints from a single frame using MediaPipe"""
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    keypoints = []

    # Pose for full body (all 33 landmarks)
    pose_results = pose_model.process(frame_rgb)
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

    # Hands
    hands_results = hands_model.process(frame_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks[:2]:  # Max 2 hands
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    # If no keypoints, return zeros (for robustness)
    if not keypoints:
        keypoints = [0.0] * 225  # 33 pose + 42 hand points * 3 = 225 dims

    # Visualize if show
    if show:
        annotated_image = frame.copy()
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Pose and Hands', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return keypoints

def read_video_and_extract_keypoints(video_path, pose_model, hands_model, target_frames=16, show=False):
    """Read video and extract keypoints for all frames"""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints_from_frame(frame, pose_model, hands_model, show=show)
        frames_keypoints.append(keypoints)
        frame_count += 1

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

# Define KeypointTransformer model (from train_mp_model.py)
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
    Input: (B, T, D) = (B, 16, 144)
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

def predict_sign_language(video_path, model_path='best_mp_model.pth', label_mapping_path='dataset/label_mapping.json', device='cuda', show=False):
    """Predict sign language from video using keypoints"""
    # Load label mapping
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Initialize model
    model = KeypointTransformer(num_classes=NUM_CLASSES, d_model=64, hidden_size=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Initialize MediaPipe
    pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # Extract keypoints
    print(f"Extracting keypoints from {video_path}...")
    frames_keypoints = read_video_and_extract_keypoints(video_path, pose_model, hands_model, TARGET_FRAMES, show=show)

    # Predict
    keypoints = torch.tensor(frames_keypoints, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, D)

    with torch.no_grad():
        outputs = model(keypoints)
        _, predicted = outputs.max(1)
        label_idx = predicted.item()
        label_name = idx_to_label[label_idx]
        confidence = torch.softmax(outputs, dim=1)[0][label_idx].item()

    if show:
        cv2.destroyAllWindows()

    return label_name, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Sign Language using MediaPipe Keypoints')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='best_mp_model.pth', help='Path to trained model')
    parser.add_argument('--labels', type=str, default='dataset/label_mapping.json', help='Path to label mapping')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--show', action='store_true', help='Show MediaPipe processing visualization')
    args = parser.parse_args()

    start_time = time()
    try:
        label, confidence = predict_sign_language(
            args.video, args.model, args.labels, args.device, args.show
        )
        print(f"Predicted Label: {label}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        end_time = time()
        print(f"Total time: {end_time - start_time:.2f} seconds")