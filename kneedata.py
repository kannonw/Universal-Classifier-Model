import os
import numpy as np
from PIL import Image
from tqdm import tqdm

root = "./datasets/KneeMRI/MRNet-v1.0"
save = "./datasets/KneeMRI"

for mode in os.listdir(root):
    actual_root = os.path.join(root, mode)

    for disease in os.listdir(actual_root):
        disease_root = os.path.join(actual_root, disease)

        for img in tqdm(os.listdir(disease_root)):
            img_root = os.path.join(disease_root, img)
            
            img_array = np.load(img_root)

            for i, img_dim in enumerate(img_array):

                img_pil = Image.fromarray(img_dim)
                img_pil.save(f"{save}/{mode}_{disease}_{img[:-4]}_{i}.png")