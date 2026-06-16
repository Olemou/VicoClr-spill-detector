# -----------------------------
# ViT Classifier
# -----------------------------
import torch
import torch.nn as nn

VIT_CONFIGS = {
    "class": {
        "num_classes": 3,
    },
    "tiny": {
        "embed_dim": 192,
    },

    "small": {
        "embed_dim": 384,
    },

    "base": {
        "embed_dim": 768,
    },
}
class ViTClassifier(nn.Module):
    def __init__(self, vit_model : nn.Module, model_size: str, freeze_vit:bool = True,dropout: float = 0.3):
        """ViT-based classifier with a trainable head and optional frozen backbone. Designed for thermal image classification.
        Args:
            vit_model: Pretrained ViT backbone (with head removed)
            model_size: ViT model size ("tiny", "small", "base")
            freeze_vit: If True, freeze the ViT backbone during training
            dropout: Dropout rate for the classification head
        """
        super().__init__()
        self.vit = vit_model
        self.vit.head = nn.Identity()  # remove original classification head
        

        cfg = VIT_CONFIGS[model_size]

        embed_dim = cfg["embed_dim"]
        num_classes = VIT_CONFIGS["class"]["num_classes"]

        # Freeze backbone
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Trainable classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),        
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)       # [B, N, D]
        cls_token = x[:, 0, :]  # CLS token
        out = self.fc(cls_token)
        return out
