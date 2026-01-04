import os
import json
from time import time
import torch
import argparse
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader

from SignLanguageModel import TARGET_FRAMES, ConvNeXtTransformer, VideoDataset


if __name__ == '__main__':
    dataset_path = 'dataset'
    save_directory = ""
    parser = argparse.ArgumentParser(description='Train Video Model')
    parser.add_argument('--resume', type=int, default=None,
                        help='Epoch number to resume training from (loads checkpoints/checkpoint_epoch_{epoch}.pth)')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                        help='Maximum number of checkpoints to keep (default: keep all)')
    args = parser.parse_args()

    label_mapping_path = os.path.join(dataset_path, 'label_mapping.json')

    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    num_labels = len(label_mapping)
    print(f"Number of labels in label_mapping.json: {num_labels}")

    model = ConvNeXtTransformer(num_classes=num_labels, hidden_size=256,
                                resnet_pretrained_weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Tạo datasets
    train_dataset_base = VideoDataset(
        model,
        f'{dataset_path}/train',
        label_to_idx_path=f'{dataset_path}/label_mapping.json',
        target_frames=TARGET_FRAMES,
        training=True  # CÓ augmentation
    )

    val_dataset_base = VideoDataset(
        model,
        f'{dataset_path}/train',
        label_to_idx_path=f'{dataset_path}/label_mapping.json',
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

    balanced_sampler = train_dataset_base.create_balanced_sampler(
        train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=balanced_sampler,  # ← THAY shuffle=True
        collate_fn=train_dataset_base.collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=val_dataset_base.collate_fn,
        num_workers=4
    )

    print(f"Train: {len(train_dataset)} (augmentation + balanced sampling)")
    print(f"Val: {len(val_dataset)} (no augmentation)")

    model = model.train_model(
        train_loader,
        val_loader,
        num_epochs=25,
        lr=1e-4,
        device='cuda',
        label_mapping_path=f'{dataset_path}/label_mapping.json',
        save_path='augmented_balanced_convnexttransformer_best_model.pth',
        resume_epoch=args.resume,
        max_checkpoints=args.max_checkpoints,
        save_directory=save_directory,
    )
