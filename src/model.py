import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint as cp
from src_utils.logging import get_logger
logger = get_logger("Model")

from timm.models.vision_transformer import trunc_normal_


# =========================================================
# ViT CONFIGS
# =========================================================

VIT_CONFIGS = {

    "tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
    },

    "small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },

    "base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
}


# =========================================================
# Attention
# =========================================================

class Attention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        qkv_bias=True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=qkv_bias
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.attn_map = None

    def forward(self, x):

        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attn = attn.softmax(dim=-1)

        self.attn_map = attn.detach()

        out = attn @ v

        out = out.transpose(1, 2)

        out = out.reshape(B, N, C)

        return self.proj(out)


# =========================================================
# Patch Dropout
# =========================================================

class PatchDropout(nn.Module):

    def __init__(
        self,
        prob=0.0,
        exclude_first_token=True,
    ):
        super().__init__()

        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):

        if (
            not self.training
            or self.prob == 0.0
        ):
            return x

        if self.exclude_first_token:

            cls_token = x[:, :1]
            x = x[:, 1:]

        else:
            cls_token = None

        B, N, C = x.shape

        keep_prob = 1.0 - self.prob

        num_keep = max(
            1,
            int(N * keep_prob)
        )

        rand = torch.randn(
            B,
            N,
            device=x.device
        )

        keep_idx = rand.topk(
            num_keep,
            dim=-1
        ).indices

        batch_idx = torch.arange(
            B,
            device=x.device
        )[:, None]

        x = x[batch_idx, keep_idx]

        if cls_token is not None:

            x = torch.cat(
                [cls_token, x],
                dim=1
            )

        return x


# =========================================================
# MLP
# =========================================================

class MLP(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.1,
    ):
        super().__init__()

        hidden_features = (
            hidden_features or in_features
        )

        out_features = (
            out_features or in_features
        )

        self.fc1 = nn.Linear(
            in_features,
            hidden_features
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_features,
            out_features
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


# =========================================================
# Transformer Block
# =========================================================

class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = Attention(
            embed_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(
                dim * mlp_ratio
            ),
            drop=drop,
        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# =========================================================
# Patch Embedding
# =========================================================

class PatchEmbed(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.num_patches = (
            img_size // patch_size
        ) ** 2

    def forward(self, x):

        x = self.proj(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x


# =========================================================
# Vision Transformer
# =========================================================

class VisionTransformer(nn.Module):

    def __init__(
        self,
        model_size="base",
        img_size=224,
        patch_size=16,
        in_chans=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_dropout_prob=0.0,
        use_checkpoint=True,
    ):
        super().__init__()

        assert model_size in VIT_CONFIGS

        cfg = VIT_CONFIGS[model_size]

        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]

        self.model_size = model_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.use_checkpoint = use_checkpoint

        # -------------------------------------------------
        # Patch embedding
        # -------------------------------------------------

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = (
            self.patch_embed.num_patches
        )

        # -------------------------------------------------
        # CLS token
        # -------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # -------------------------------------------------
        # Position embedding
        # -------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim
            )
        )

        self.pos_drop = nn.Dropout(0.0)

        # -------------------------------------------------
        # Patch dropout
        # -------------------------------------------------

        self.patch_dropout = (
            PatchDropout(
                patch_dropout_prob
            )
            if patch_dropout_prob > 0
            else nn.Identity()
        )

        # -------------------------------------------------
        # Transformer blocks
        # -------------------------------------------------

        self.blocks = nn.ModuleList([

            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )

            for _ in range(depth)

        ])

        # -------------------------------------------------
        # Final norm
        # -------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        # -------------------------------------------------
        # Projection head
        # -------------------------------------------------

        self.head = nn.Sequential(
              nn.Linear(embed_dim, embed_dim),
              nn.LayerNorm(embed_dim),
              nn.GELU(),
              nn.Dropout(0.1),

              nn.Linear(embed_dim, 256),
              nn.LayerNorm(256),
          )
        # -------------------------------------------------
        # Initialization
        # -------------------------------------------------

        trunc_normal_(
            self.cls_token,
            std=0.02
        )

        trunc_normal_(
            self.pos_embed,
            std=0.02
        )

        self.apply(
            self._init_weights
        )

    # =====================================================
    # Initialization
    # =====================================================

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):

            trunc_normal_(
                m.weight,
                std=0.02
            )

            if m.bias is not None:

                nn.init.zeros_(
                    m.bias
                )

        elif isinstance(
            m,
            nn.LayerNorm
        ):

            nn.init.ones_(
                m.weight
            )

            nn.init.zeros_(
                m.bias
            )

    # =====================================================
    # Attention extraction
    # =====================================================

    @torch.no_grad()
    def get_attentions(
        self,
        images,
        stack=True,
        eval_mode=True,
    ):

        was_training = self.training

        if eval_mode:
            self.eval()

        _ = self(images)

        attentions = [

            blk.attn.attn_map

            for blk in self.blocks

        ]

        if stack:

            attentions = torch.stack(
                attentions,
                dim=0
            )

        if was_training and eval_mode:
            self.train()

        return attentions

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        B = x.shape[0]

        # patch embedding
        x = self.patch_embed(x)

        # cls token
        cls_token = self.cls_token.expand(
            B,
            -1,
            -1
        )

        x = torch.cat(
            [cls_token, x],
            dim=1
        )

        # positional embedding
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # patch dropout
        x = self.patch_dropout(x)

        # transformer blocks
        for block in self.blocks:

            if (
                self.use_checkpoint
                and self.training
            ):

                x = cp.checkpoint(
                    block,
                    x,use_reentrant=False
                )

            else:
                x = block(x)

        # final norm
        x = self.norm(x)

        # remove cls token
        x = x[:, 1:, :]

        return self.head(x)


# =========================================================
# PRETRAINED LOADER
# =========================================================
def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["encoder"])
    else:
        model.load_state_dict(checkpoint["encoder"])

    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint.get("epoch", 0) + 1

    logger.info(
        f"✅ Checkpoint loaded successfully from epoch {start_epoch - 1}"
    )

    return start_epoch
# ========================================================= Default Pretrained Loader

def default_load_pretrained_weights(
    custom_model,
    device="cuda",
):
    """
    Automatically loads:
        tiny  -> vit_tiny_patch16_224
        small -> vit_small_patch16_224
        base  -> vit_base_patch16_224
    """

    size_to_timm = {

        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224",
        "base": "vit_base_patch16_224",
    }

    timm_model_name = size_to_timm[
        custom_model.model_size
    ]

    custom_model = custom_model.to(device)

    pretrained_model = timm.create_model(
        timm_model_name,
        pretrained=True,
    ).to(device)

    pretrained_dict = (
        pretrained_model.state_dict()
    )

    custom_dict = custom_model.state_dict()

    compatible_weights = {}

    for k, v in pretrained_dict.items():

        if (
            k in custom_dict
            and v.shape == custom_dict[k].shape
        ):

            compatible_weights[k] = v

    custom_dict.update(
        compatible_weights
    )

    custom_model.load_state_dict(
        custom_dict,
        strict=False
    )

    print(
        f"\nLoaded pretrained {custom_model.model_size} ViT"
    )

    print(
        f"Matched weights: {len(compatible_weights)}"
    )

    return custom_model


# =========================================================
# EXAMPLE
# =========================================================

# tiny
model = VisionTransformer(
    model_size="tiny",
    patch_dropout_prob=0.1,
)

# small
# model = VisionTransformer(
#     model_size="small"
# )

# base
# model = VisionTransformer(
#     model_size="base"
# )

