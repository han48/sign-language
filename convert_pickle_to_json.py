import pickle
import json

with open('dataset/label_mapping.pkl', 'rb') as f:
    data = pickle.load(f)

with open('dataset/label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)