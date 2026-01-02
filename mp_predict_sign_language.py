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
    """Extract arm and hand keypoints from a single frame using MediaPipe
    
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

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Extract pose keypoints (indices 0-98)
    pose_results = pose_model.process(frame_rgb)
    if pose_results.pose_landmarks:
        for i, lm in enumerate(pose_results.pose_landmarks.landmark):
            keypoints[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

    # Extract hand keypoints (indices 99-224: hand1 99-161, hand2 162-224)
    hand_start_idx = 99
    hands_results = hands_model.process(frame_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
            for i, lm in enumerate(hand_landmarks.landmark):
                start = hand_start_idx + hand_idx * 63 + i * 3
                keypoints[start:start+3] = [lm.x, lm.y, lm.z]

    # Visualize if show
    if show:
        annotated_image = frame.copy()
        if pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hands_results and hands_results.multi_hand_landmarks:
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

def predict_sign_language(video_path, model_path='models\mp_vls.pth', label_mapping_path='dataset/label_mapping.json', device='cuda', show=False):
    """Predict sign language from video using keypoints"""
    # Load label mapping
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Initialize model
    model = KeypointTransformer(num_classes=NUM_CLASSES, d_model=64, hidden_size=256)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
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
    parser.add_argument('--model', type=str, default='models\mp_vls.pth', help='Path to trained model')
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