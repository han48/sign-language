import argparse
from SignLanguageModel import KeypointTransformer, create_dataset_statistics, create_label_mapping, organize_dataset


if __name__ == '__main__':
    # Aggregate command-line arguments
    # Action
    parser = argparse.ArgumentParser("Sign Language Model Utilities")
    parser.add_argument('--action', type=str, required=True,
                        choices=['organize_dataset', 'create_label_mapping',
                                 'create_dataset_statistics', 'preprocess_keypoints'],
                        help="Action to perform")
    args = parser.parse_args()

    if args.action == 'organize_dataset':
        organize_dataset()
    elif args.action == 'create_label_mapping':
        create_label_mapping()
    elif args.action == 'create_dataset_statistics':
        create_dataset_statistics()
    elif args.action == 'preprocess_keypoints':
        model = KeypointTransformer()
        # Preprocess keypoints for train dataset
        print("Preprocessing keypoints for train dataset...")
        model.preprocess_keypoints(
            'dataset/train', 'dataset/label_mapping.json',
            show=True, force_recreate=False, multiple_mp=True, num_workers=20
        )
    else:
        print("Unknown action.")
