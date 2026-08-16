import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Single Residual Block for PixelSiliconNet.

    Architecture:
    1. Conv2d(64, 64, kernel_size=3, padding=1)
    2. ReLU activation
    3. Conv2d(64, 64, kernel_size=3, padding=1)
    4. Residual skip connection (output = input + residual)
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute residual path
        residual = self.conv2(self.relu(self.conv1(x)))
        # Element-wise shortcut addition
        return x + residual


class PixelSiliconNet(nn.Module):
    """
    Lightweight Grayscale Image Restoration and 2x Super-Resolution CNN (PixelSiliconNet).

    Input Tensor:  [B, 1, 128, 128]
    Output Tensor: [B, 1, 256, 256]

    Detailed Architectural Steps:
    ------------------------------
    1. Input Feature Extraction:
       Conv2d(1, 64, kernel_size=3, padding=1)

    2. Residual Trunk:
       8 Residual Blocks stacked sequentially.
       Each block: Conv2d(64,64,3) -> ReLU -> Conv2d(64,64,3) + Skip Connection

    3. Global Residual Connection:
       Adds the initial feature map directly to the output of the complete residual trunk.

    4. 2x Upsampling:
       Conv2d(64, 64*4, kernel_size=3, padding=1) -> PixelShuffle(upscale_factor=2) -> ReLU

    5. Reconstruction Layer:
       Conv2d(64, 1, kernel_size=3, padding=1)

    6. Output Activation:
       Sigmoid activation mapping predictions into [0, 1] range.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 8,
        upscale_factor: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        # Step 1: Input feature extraction layer (1 -> 64 channels)
        self.input_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)

        # Step 2: Residual trunk (8 Residual Blocks)
        self.res_trunk = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])

        # Step 4: Upsampling layer (64 -> 64*4 channels followed by PixelShuffle and ReLU)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )

        # Step 5: Reconstruction layer (64 -> 1 output channel)
        self.reconstruction = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)

        # Step 6: Output activation for [0, 1] range constraint
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PixelSiliconNet.

        Args:
            x (torch.Tensor): Input LR image tensor of shape [B, 1, 128, 128]

        Returns:
            torch.Tensor: Restored 2x SR image tensor of shape [B, 1, 256, 256]
        """
        # Step 1: Input feature extraction -> [B, 64, 128, 128]
        f_in = self.input_conv(x)

        # Step 2: Pass through complete 8-block residual trunk -> [B, 64, 128, 128]
        f_trunk = self.res_trunk(f_in)

        # Step 3: Global residual connection around the complete residual trunk
        f_global = f_trunk + f_in

        # Step 4: 2x Upsampling via PixelShuffle -> [B, 64, 256, 256]
        f_up = self.upsample(f_global)

        # Step 5 & 6: Reconstruction convolution & Sigmoid output activation -> [B, 1, 256, 256]
        output = self.sigmoid(self.reconstruction(f_up))

        return output


def print_trainable_parameters(model: nn.Module) -> int:
    """
    Calculates and prints the total number of trainable parameters in the model.

    Args:
        model (nn.Module): PyTorch model instance.

    Returns:
        int: Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters Count: {total_params:,} ({total_params / 1e6:.4f} M)")
    return total_params


if __name__ == "__main__":
    # Self-test block: Instantiate PixelSiliconNet and perform forward pass on [1, 1, 128, 128] input
    print("=" * 60)
    print("            PIXELSILICONNET SELF-TEST SUMMARY           ")
    print("=" * 60)

    # Create model instance
    model = PixelSiliconNet(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=8,
        upscale_factor=2
    )

    # Print trainable parameters
    param_count = print_trainable_parameters(model)

    # Create random test input with shape [1, 1, 128, 128]
    test_input = torch.randn(1, 1, 128, 128)

    # Execute forward pass
    model.eval()
    with torch.no_grad():
        test_output = model(test_input)

    # Print input shape, output shape, and output value range
    print("-" * 60)
    print(f"Input Shape:   {list(test_input.shape)}")
    print(f"Output Shape:  {list(test_output.shape)}")
    print(f"Output Range:  [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
    print("=" * 60)

    # Assert shape correctness
    assert test_output.shape == (1, 1, 256, 256), f"Unexpected output shape: {test_output.shape}"
    assert 0.0 <= test_output.min().item() and test_output.max().item() <= 1.0, "Output out of [0, 1] range!"
    print("\nSelf-test passed successfully!")
