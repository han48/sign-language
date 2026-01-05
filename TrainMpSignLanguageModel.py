import os
import json
import cv2
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from SignLanguageModel import TARGET_FRAMES, KeypointDataset, KeypointTransformer


if __name__ == '__main__':
    dataset_path = 'dataset'
    save_directory = ""
    parser = argparse.ArgumentParser(
        description='Train MediaPipe Keypoint Model')
    parser.add_argument('--show', action='store_true',
                        help='Show MediaPipe processing visualization')
    parser.add_argument('--force-recreate', action='store_true',
                        help='Force recreate cache files even if they exist')
    parser.add_argument('--resume', type=int, default=None,
                        help='Epoch number to resume training from (loads mp_checkpoints/mp_checkpoint_epoch_{epoch}.pth)')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                        help='Maximum number of checkpoints to keep (default: keep all)')
    args = parser.parse_args()

    label_mapping_path = os.path.join(dataset_path, 'label_mapping.json')

    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    num_labels = len(label_mapping)
    print(f"Number of labels in label_mapping.json: {num_labels}")

    model = KeypointTransformer(
        num_classes=num_labels, d_model=64, hidden_size=256)

    # Preprocess keypoints for train dataset
    print("Preprocessing keypoints for train dataset...")
    model.preprocess_keypoints('dataset/train', 'dataset/label_mapping.json',
                               show=args.show, force_recreate=args.force_recreate, multiple_mp=True)

    # Preprocess keypoints for public test
    print("Preprocessing keypoints for public test...")
    model.preprocess_keypoints('dataset/public_test', 'dataset/label_mapping.json',
                               show=args.show, force_recreate=args.force_recreate)

    # Preprocess keypoints for private test
    print("Preprocessing keypoints for private test...")
    model.preprocess_keypoints('dataset/private_test', 'dataset/label_mapping.json',
                               show=args.show, force_recreate=args.force_recreate)

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

    balanced_sampler = model.create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Larger batch for keypoints
        sampler=balanced_sampler,
        collate_fn=model.collate_fn_keypoints,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=model.collate_fn_keypoints,
        num_workers=4
    )

    print(f"Train: {len(train_dataset)} (balanced sampling)")
    print(f"Val: {len(val_dataset)}")

    model = model.train_keypoint_model(
        train_loader,
        val_loader,
        num_epochs=100,
        lr=1e-4,
        device='cuda',
        label_mapping_path=f'{dataset_path}/label_mapping.json',
        save_path='best_mp_model.pth',
        resume_epoch=args.resume,
        max_checkpoints=args.max_checkpoints,
        save_directory=save_directory,
    )

    # Export public result
    model.evaluate_keypoints(
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
    model.evaluate_keypoints(
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
