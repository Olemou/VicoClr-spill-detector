import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import torch

class ThermalAugmentation:
    def __init__(self, image_size=224, p_geo=0.5, p_thermo=0.7, p_occl=0.4):
        self.image_size = image_size
        self.p_geo = p_geo
        self.p_thermo = p_thermo
        self.p_occl = p_occl

        self.transform = self.build_transform()

    # =========================================================
    # Thermal photometric variations
    # =========================================================
    def thermal_contrast(self, x):
        x = np.array(x).astype(np.float32)
        alpha = np.random.uniform(0.8, 1.2)
        x = alpha * x
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))

    def brightness_shift(self, x):
        x = np.array(x).astype(np.float32)
        shift = np.random.uniform(-20, 20)
        x = x + shift
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))

    # =========================================================
    # Occlusion (thermal noise / sensor drop)
    # =========================================================
    def thermal_erase(self, x):
        x = np.array(x)
        h, w = x.shape[:2]

        mask_w, mask_h = int(w * 0.2), int(h * 0.2)

        for _ in range(10):
            x1 = np.random.randint(0, w - mask_w)
            y1 = np.random.randint(0, h - mask_h)

            x[y1:y1+mask_h, x1:x1+mask_w] = 0
            break

        return Image.fromarray(x)
    
    def visualize_all(self, img):
    
        mean = np.array([0.24, 0.24, 0.24])
        std = np.array([0.07, 0.07, 0.07])

        # -------------------------------------------------
        # raw augmentations (PIL outputs)
        # -------------------------------------------------
        imgs = {
            "original": np.array(img),
            "thermal_contrast": np.array(self.thermal_contrast(img)),
            "brightness_shift": np.array(self.brightness_shift(img)),
            "thermal_erase": np.array(self.thermal_erase(img.copy())),
        }

        # -------------------------------------------------
        # full pipeline (tensor → need denorm)
        # -------------------------------------------------
        full = self.transform(img)  # CHW tensor

        if isinstance(full, torch.Tensor):
            full = full.detach().cpu().numpy()

            # denormalize
            for c in range(full.shape[0]):
                full[c] = full[c] * std[c] + mean[c]

            full = np.transpose(full, (1, 2, 0))
            full = np.clip(full, 0, 1)

        imgs["full_pipeline"] = full
        return imgs
    # =========================================================
    # Transform pipeline
    # =========================================================
    def build_transform(self):

        geo = transforms.RandomApply([
            transforms.RandomAffine(
                degrees=8,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            )
        ], p=self.p_geo)

        thermal_noise = transforms.RandomApply([
            transforms.Lambda(self.thermal_contrast),
        ], p=self.p_thermo)

        brightness = transforms.RandomApply([
            transforms.Lambda(self.brightness_shift),
        ], p=self.p_thermo)

        occlusion = transforms.RandomApply([
            transforms.Lambda(self.thermal_erase),
        ], p=self.p_occl)

        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),

            # ---- Geometry (light) ----
            geo,
            transforms.RandomHorizontalFlip(p=0.5),

            # ---- Thermal appearance ----
            thermal_noise,
            brightness,

            # ---- Robustness ----
            occlusion,
            

            # ---- Final cleanup ----
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.24, 0.24, 0.24],
                std=[0.07, 0.07, 0.07]
            )
        ])

    def __call__(self, img):
        return self.transform(img)