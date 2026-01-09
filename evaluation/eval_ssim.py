import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def load_gray(path):
    img = Image.open(path).convert("L").resize((512, 512))
    return np.array(img)

def eval_folder(folder, label):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    scores = []
    
    for i in range(len(files) - 1):
        img1 = load_gray(os.path.join(folder, files[i]))
        img2 = load_gray(os.path.join(folder, files[i + 1]))
        
        score = ssim(img1, img2, data_range=255)
        scores.append(score)
        
    print(f"{label} | Mean SSI: {np.mean(scores):4f}")
    return scores

if __name__ == "__main__":
    base = "experiments/inference/baseline"
    finetuned = "experiments/inference/finetuned"
    tile = "experiments/tile/finetuned"
    
    eval_folder(base, "Baseline")
    eval_folder(finetuned, "LoRA")
    eval_folder(tile, "Tiled")