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

# Constants
NUM_CLASSES = 100
TARGET_FRAMES = 16  # number of frames per video

    # Read video frames using OpenCV with fps detection
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback
    frames = []
    with tqdm(desc="Reading video frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            pbar.update(1)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    frames = torch.from_numpy(np.stack(frames, axis=0))
    return frames, fps

# Define ConvNeXtTransformer model
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

# Preprocessing functions from VideoDataset
class VideoPreprocessor:
    def __init__(self, target_frames=32, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.target_frames = target_frames
        self.mean = mean
        self.std = std

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
        frames = read_video(video_path)
        frames = self._downsample_frames(frames)
        frames = self._normalize(frames)
        return frames.unsqueeze(0)  # Add batch dimension

# Function to predict sign language from long video (sentence)
def predict_sign_language_sentence(video_path, model_path='models/abc_vsl.pth', label_mapping_path='dataset/label_mapping.pkl', device=None, window_size=16, stride=8, prediction_method='consecutive', confidence_threshold=0.0, block_durations=None, target_fps=None, block_duration_for_summary=1, show=False):
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load label mapping (prefer JSON, fallback to PKL)
    label_mapping_file = label_mapping_path
    if not os.path.exists(label_mapping_file):
        if label_mapping_file.endswith('.json'):
            pkl_file = label_mapping_file.replace('.json', '.pkl')
            if os.path.exists(pkl_file):
                label_mapping_file = pkl_file
            else:
                raise FileNotFoundError(f"Neither {label_mapping_file} nor {pkl_file} found")
    
    if label_mapping_file.endswith('.json'):
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
    else:
        with open(label_mapping_file, 'rb') as f:
            label_mapping = pickle.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Initialize model
    model = ConvNeXtTransformer(num_classes=NUM_CLASSES, hidden_size=256, resnet_pretrained_weights=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Read and preprocess entire video
    frames, fps = read_video(video_path)
    num_frames = frames.shape[0]
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
        print(f"Loaded {len(frames_display)} frames for display")
        
        # Get screen size
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        print(f"Screen size: {screen_width}x{screen_height}")
        
        # Create windows with minimal controls
        # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Info', cv2.WINDOW_GUI_NORMAL)
        
    # Resample frames if target_fps is set
    if target_fps is not None and target_fps != fps:
        step = int(fps / target_fps)
        if step > 1:
            frames = frames[::step]
            num_frames = frames.shape[0]
            fps = target_fps
            print(f"Resampled to {num_frames} frames at {fps} FPS")

    # Normalize frames
    preprocessor = VideoPreprocessor(target_frames=window_size)
    frames = preprocessor._normalize(frames)  # Normalize without downsampling

    predictions = []

    # Sliding window prediction with progress bar
    total_windows = len(range(0, num_frames - window_size + 1, stride))
    with tqdm(total=total_windows, desc="Processing video windows") as pbar:
        for start in range(0, num_frames - window_size + 1, stride):
            end = start + window_size
            window_frames = frames[start:end]  # (window_size, C, H, W)
            window_frames = window_frames.unsqueeze(0).to(device)  # Add batch dim

            with torch.no_grad():
                outputs = model(window_frames)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
                label_idx = predicted.item()
                label_name = idx_to_label[label_idx]
                conf_value = confidence.item()
                predictions.append((label_name, conf_value))
            
            if show and frames_display is not None and end - 1 < len(frames_display):
                display_frame = frames_display[end - 1].copy()
                cv2.imshow('Frame', display_frame)
                
                # Create a separate window for text info
                info_image = np.zeros((100, 400, 3), dtype=np.uint8)
                text = f"Processing frame index: {start}-{end}"
                cv2.putText(info_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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

    # Process predictions based on method
    if prediction_method == 'consecutive':
        # Remove duplicates and return unique predictions (simple approach)
        unique_predictions = []
        for pred, conf in predictions:
            if conf >= confidence_threshold and (not unique_predictions or pred != unique_predictions[-1][0]):
                unique_predictions.append((pred, conf))
    elif prediction_method == 'majority':
        # Filter by confidence
        filtered_predictions = [(p, c) for p, c in predictions if c >= confidence_threshold]
        if not filtered_predictions:
            filtered_predictions = predictions  # Fallback if all filtered out
        # Majority vote: count frequencies and select based on count, keeping order of first appearance
        label_counts = Counter([p for p, c in filtered_predictions])
        seen = OrderedDict()
        for p, c in filtered_predictions:
            if p not in seen:
                seen[p] = label_counts[p]
        # Sort by count descending, then by first appearance order
        sorted_labels = sorted(seen.items(), key=lambda x: (-x[1], list(seen.keys()).index(x[0])))
        total = len(filtered_predictions)
        unique_predictions = [(p, count / total) for p, count in sorted_labels]
    elif prediction_method == 'smooth':
        # Filter by confidence
        filtered_predictions = [(p, c) for p, c in predictions if c >= confidence_threshold]
        if not filtered_predictions:
            filtered_predictions = predictions
        # Simple smoothing: group every 5 predictions, take majority per group
        group_size = 5
        smoothed = []
        for i in range(0, len(filtered_predictions), group_size):
            group = filtered_predictions[i:i+group_size]
            if group:
                label_counts = Counter([p for p, c in group])
                most_common = label_counts.most_common(1)[0][0]
                avg_conf = sum(c for p, c in group if p == most_common) / len([p for p, c in group if p == most_common])
                smoothed.append((most_common, avg_conf))
        unique_predictions = smoothed
    elif prediction_method == 'temporal':
        # Time-based blocks: multiple durations (1,2,3 seconds), 0.5 second shift, collect all predictions
        if block_durations is None:
            block_durations = [1, 2, 3]
        shift_duration = 0.5  # seconds
        shift_frames = int(fps * shift_duration)
        if shift_frames == 0:
            shift_frames = 15
        all_predictions = []
        total_blocks = sum((num_frames - int(fps * d) + 1) // shift_frames + 1 for d in block_durations if int(fps * d) > 0)
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
                        window_frames = preprocessor._downsample_frames(window_frames)
                    window_frames = window_frames.unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(window_frames)
                        probs = F.softmax(outputs, dim=1)
                        confidence, predicted = probs.max(1)
                        label_idx = predicted.item()
                        label_name = idx_to_label[label_idx]
                        conf_value = confidence.item()
                        if conf_value >= confidence_threshold:
                            all_predictions.append((label_name, conf_value, start_time_sec, end_time_sec, duration))
                    
                    if show and frames_display is not None and end - 1 < len(frames_display):
                        display_frame = frames_display[end - 1].copy()
                        cv2.imshow('Frame', display_frame)
                        
                        # Create a separate window for text info
                        info_image = np.zeros((100, 600, 3), dtype=np.uint8)
                        text = f"Processing block: {start}-{end} (frames), {start_time_sec:.1f}-{end_time_sec:.1f}s"
                        cv2.putText(info_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
                    
                    pbar.set_postfix(label=f"{label_name} ({conf_value:.2f})", time=f"{start_time_sec:.1f}-{end_time_sec:.1f}s")
                    pbar.update(1)
        # Print all predictions
        print("All predictions from temporal blocks:")
        for i, (label, conf, start_t, end_t, dur) in enumerate(all_predictions):
            print(f"{i+1}: {label} ({conf:.2f}) [{start_t:.1f}-{end_t:.1f}s] ({dur}s block)")
        # Summarize best result using majority vote with average confidence from specified block duration
        if all_predictions:
            filtered_predictions = [pred for pred in all_predictions if pred[4] == block_duration_for_summary]
            if filtered_predictions:
                label_counts = Counter([p for p, c, _, _, _ in filtered_predictions])
                label_avg_conf = {}
                for p in label_counts:
                    confs = [c for pred, c, _, _, _ in filtered_predictions if pred == p]
                    label_avg_conf[p] = sum(confs) / len(confs)
                seen = OrderedDict()
                for p, c, s, e, d in filtered_predictions:
                    if p not in seen:
                        seen[p] = (label_counts[p], s)  # count and first start time
                # Find the maximum count
                if seen:
                    max_count = max(v[0] for v in seen.values())
                    # Filter labels with count >= 2 (to keep significant ones)
                    candidate_labels = [(p, v) for p, v in seen.items() if v[0] >= 2]
                    # Sort by first appearance time ascending, then by average confidence descending
                    sorted_labels = sorted(candidate_labels, key=lambda x: (x[1][1], -label_avg_conf[x[0]]))
                    unique_predictions = [(p, label_avg_conf[p]) for p, _ in sorted_labels]
                else:
                    unique_predictions = []
                print(f"Best summary (majority vote with avg confidence from {block_duration_for_summary}s blocks): {' '.join([f'{p}({c:.2f})' for p, c in unique_predictions])}")
            else:
                unique_predictions = []
                print(f"No predictions from {block_duration_for_summary}s blocks.")
        else:
            unique_predictions = []
    else:
        raise ValueError("prediction_method must be 'consecutive', 'majority', 'smooth', or 'temporal'")

    if show:
        cv2.destroyAllWindows()

    return unique_predictions

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sign language from video using trained model.")
    parser.add_argument("video_path", help="Path to the sign language video file.")
    parser.add_argument("--model_path", default="models/abc_vsl.pth", help="Path to the trained model file.")
    parser.add_argument("--label_mapping_path", default="dataset/label_mapping.json", help="Path to the label mapping file (JSON preferred, PKL fallback).")
    parser.add_argument("--device", default=None, help="Device to run the model on (cuda, cpu, or auto-detect if not specified).")
    parser.add_argument("--window_size", type=int, default=16, help="Window size for sliding window prediction.")
    parser.add_argument("--stride", type=int, default=8, help="Stride for sliding window prediction.")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Minimum confidence threshold to consider predictions (0.0 to 1.0).")
    parser.add_argument("--target_fps", type=float, default=10, help="Target FPS to resample video frames to match training data (e.g., 10.0).")
    parser.add_argument("--prediction_method", choices=['consecutive', 'majority', 'smooth', 'temporal'], default='consecutive', help="Method to process predictions: 'consecutive' for unique consecutive labels, 'majority' for majority vote, 'smooth' for grouped smoothing, 'temporal' for time-based blocks.")
    parser.add_argument("--block_duration_for_summary", type=int, default=1, help="Block duration in seconds to use for summary in temporal method (1, 2, or 3).")
    parser.add_argument("--show", action='store_true', help="Show frames during processing with frame index and block information.")

    args = parser.parse_args()

    try:
        start_time = time()
        predicted_labels = predict_sign_language_sentence(args.video_path, args.model_path, args.label_mapping_path, args.device, args.window_size, args.stride, args.prediction_method, args.confidence_threshold, None, args.target_fps, args.block_duration_for_summary, args.show)
        if args.prediction_method != 'temporal':
            print(f"Predicted sign language sentence: {' '.join([f'{p}({c:.2f})' for p, c in predicted_labels])}")
        end_time = time()
        print(f"Prediction took {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()