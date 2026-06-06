import torch

from sckatisation_net.config import ModelConfig
from sckatisation_net.model import SCKATisionNet, KANLayer


def test_forward_pass_small():
    cfg = ModelConfig(image_size=64, patch_size=16, embed_dim=32, depth=1, num_heads=4, kan_hidden_dim=64, num_classes=5)
    model = SCKATisionNet(cfg)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_kan_layer_shape():
    layer = KANLayer(8, 4, grid_size=5)
    x = torch.randn(3, 7, 8)
    y = layer(x)
    assert y.shape == (3, 7, 4)
