from time import time
import argparse

from SignLanguageModel import KeypointTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict Sign Language using MediaPipe Keypoints with Temporal Blocks')
    parser.add_argument(
        "video_path", help="Path to the sign language video file.")
    parser.add_argument('--model_path', type=str,
                        default='models/mp_vls.pth', help='Path to trained model')
    parser.add_argument('--device', type=str,
                        default='cuda', help='Device to run on')
    parser.add_argument('--confidence_threshold', type=float,
                        default=0.0, help='Minimum confidence threshold')
    parser.add_argument("--target_fps", type=float, default=16,
                        help="Target FPS to resample video frames to match training data (e.g., 10.0).")
    parser.add_argument('--block_duration_for_summary', type=int,
                        default=1, help='Block duration for summary (1, 2, or 3)')
    parser.add_argument('--show', action='store_true',
                        help='Show MediaPipe processing visualization')
    parser.add_argument("--debug", action='store_true',
                        help="Show debug information.")
    args = parser.parse_args()

    model = KeypointTransformer.load_model(
        model_path=args.model_path, device=args.device, debug=args.debug)

    start_time = time()
    results = model.predict_sign_language(
        args.video_path, args.device, args.show,
        block_durations=[1, 2, 3],
        confidence_threshold=args.confidence_threshold,
        block_duration_for_summary=args.block_duration_for_summary,
        target_fps=args.target_fps, 
        debug=args.debug,
    )
    print(
        f"\nFinal Predicted Labels: {' '.join([f'{label}({conf:.4f})' for label, conf in results])}")

    end_time = time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
