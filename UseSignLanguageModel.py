from time import time
import argparse

from SignLanguageModel import ConvNeXtTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict sign language from video using trained model.")
    parser.add_argument(
        "video_path", help="Path to the sign language video file.")
    parser.add_argument("--model_path", default="models/abc_vsl.pth",
                        help="Path to the trained model file.")
    parser.add_argument("--device", default=None,
                        help="Device to run the model on (cuda, cpu, or auto-detect if not specified).")
    parser.add_argument("--window_size", type=int, default=16,
                        help="Window size for sliding window prediction.")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride for sliding window prediction.")
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="Minimum confidence threshold to consider predictions (0.0 to 1.0).")
    parser.add_argument("--target_fps", type=float, default=16,
                        help="Target FPS to resample video frames to match training data (e.g., 10.0).")
    parser.add_argument("--block_duration_for_summary", type=int, default=1,
                        help="Block duration in seconds to use for summary in temporal method (1, 2, or 3).")
    parser.add_argument("--show", action='store_true',
                        help="Show frames during processing with frame index and block information.")
    parser.add_argument("--debug", action='store_true',
                        help="Show debug information.")

    args = parser.parse_args()

    start_time = time()
    model = ConvNeXtTransformer.load_model(
        model_path=args.model_path, device=args.device)

    file_name = "prediction_status.vsl"

    def fn_push(status):
        with open(file_name, "w", encoding="utf-8") as f:
            text = ""
            confidence = 0
            if len(status) > 3:
                unique_predictions = model.summarize_best_result(status[3], args.block_duration_for_summary)
                text = " ".join([k for k, _ in unique_predictions])
                confidence = sum(v for _, v in unique_predictions) / len(unique_predictions) if unique_predictions else float('inf')
            content = f"{model.predict_steps.index(status[0]) + 1},{len(model.predict_steps)},{status[0]},{status[1]},{status[2]},{text},{confidence:.2f}"
            f.write(content)

    text, confidence, predicted_labels = model.predict_sign_language_sentence(
        args.video_path, fn_push=fn_push, window_size=args.window_size, stride=args.stride, confidence_threshold=args.confidence_threshold, block_durations=None, target_fps=args.target_fps, block_duration_for_summary=args.block_duration_for_summary, show=args.show, debug=args.debug)
    print(text)
    print(confidence)
    print(f"{' '.join([f'{p}({c:.2f})' for p, c in predicted_labels])}")
    end_time = time()
    print(f"Prediction took {end_time - start_time:.2f} seconds.")
