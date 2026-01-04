# RUN CLI

```
python predict_sign_language.py "dataset\input.mp4" --device cuda --prediction_method temporal --confidence_threshold 0.5 --show

python abc_predict_sign_language.py "dataset\input.mp4" --device cuda --confidence_threshold 0.5 --show

python mp_predict_sign_language.py --video "dataset\input.mp4" --device cuda --confidence_threshold 0.5 --show
```