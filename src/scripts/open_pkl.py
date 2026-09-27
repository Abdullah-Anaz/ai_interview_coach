import pickle
from pathlib import Path

file_path = Path("src/data/chalearn/annotations/test-annotation-e/annotation_test.pkl")

with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print(f"Root data type: {type(data)}")

if isinstance(data, dict):
    keys = list(data.keys())
    print(f"Total keys at root: {len(keys)}")
    print(f"First 5 keys: {keys[:5]}")
    
    if keys:
        first_key = keys[0]
        sample_value = data[first_key]
        print(f"\nSample value type for key '{first_key}': {type(sample_value)}")
        
        if isinstance(sample_value, dict):
            print(f"Keys inside nested dictionary: {list(sample_value.keys())[:5]}")
            nested_key = list(sample_value.keys())[0]
            print(f"Sample text: {sample_value[nested_key]}")
        else:
            print(f"Sample text: {sample_value}")