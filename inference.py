import argparse
import torch
import traceback
from PIL import Image
from model_arch import PneumoniaModel, get_test_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# 1. Load Model and Transforms
try:
    model = PneumoniaModel(2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
except Exception as e:
    print(f"Warning: Could not load model from '{MODEL_PATH}' ({e}). Classification may fail.")
    model = None

test_transform = get_test_transform()

def classify_xray(img):
    if img is None:
        return "No image provided"
    if model is None:
        return "Model not loaded"
    try:
        # Preprocessing
        img = img.convert("RGB")
        x = test_transform(img).unsqueeze(0)
        
        # Inference
        x = x.to(DEVICE)
        
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, 1)[0]
            idx = prob.argmax().item()
            
        classes = ['Normal', 'Pneumonia']
        
        return f"{classes[idx]} ({prob[idx]:.4f})"
    except Exception:
        return traceback.format_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pneumonia X-ray Classifier CLI")
    parser.add_argument("image_path", type=str, help="Path to the chest X-ray image file")
    args = parser.parse_args()
    
    try:
        img = Image.open(args.image_path)
        print(f"Classifying '{args.image_path}' using device: {DEVICE}...")
        result = classify_xray(img)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error opening image: {e}")