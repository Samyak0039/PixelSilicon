import os
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class KLADataset(Dataset):
    """
    PyTorch Dataset for the KLA Semiconductor Wafer Inspection Dataset.

    Paired Dataset Directory Layout:
        dataset/KLA/train/train/NoisyLR/ (Input: 128x128 .npy)
        dataset/KLA/train/train/GT/      (Target: 256x256 .npy)

    Implementation Details & Design Choices:
    ----------------------------------------
    1. Paired Matching: Matches files by identical filename across NoisyLR/ and GT/.
    2. macOS Cleanup: Explicitly ignores '.DS_Store', '_MACOSX', and non-.npy files.
    3. Tensor Conversion & Shapes: Converts arrays to torch.float32 and adds channel dimension:
       - NoisyLR Input:  [1, 128, 128]
       - GT Target:     [1, 256, 256]
    4. GT Normalization: Clamps Ground Truth target values consistently to [0, 1].
    5. Noise Preservation: Preserves raw noisy values in NoisyLR without blind clipping to [0, 1].
       Optical inspection sensor noise (additive Gaussian + multiplicative speckle) produces values
       slightly below 0.0 or above 1.0. Blind clipping distorts physical noise statistics needed
       by restoration models to learn the true inverse degradation mapping.
    """

    def __init__(
        self,
        root_dir: str = "dataset/KLA/train/train",
        noisy_dir: Optional[str] = None,
        gt_dir: Optional[str] = None,
        normalize_gt: bool = True
    ):
        """
        Args:
            root_dir (str): Root path containing 'NoisyLR' and 'GT' subdirectories.
            noisy_dir (str, optional): Custom path to NoisyLR folder. Overrides root_dir/NoisyLR.
            gt_dir (str, optional): Custom path to GT folder. Overrides root_dir/GT.
            normalize_gt (bool): If True, clamps GT target values to [0.0, 1.0]. Default is True.
        """
        super().__init__()

        # Resolve directory paths
        if noisy_dir is None:
            noisy_dir = os.path.join(root_dir, "NoisyLR")
        if gt_dir is None:
            gt_dir = os.path.join(root_dir, "GT")

        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.normalize_gt = normalize_gt

        # Validate existence of directories
        if not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"NoisyLR directory does not exist: {self.noisy_dir}")
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"GT directory does not exist: {self.gt_dir}")

        # Helper function to filter valid .npy files while ignoring .DS_Store, _MACOSX, and hidden files
        def is_valid_file(filename: str) -> bool:
            if not filename.endswith(".npy"):
                return False
            if filename.startswith(".") or "_MACOSX" in filename or "DS_Store" in filename:
                return False
            return True

        # Read directory listings and filter files
        noisy_files = {f for f in os.listdir(self.noisy_dir) if is_valid_file(f)}
        gt_files = {f for f in os.listdir(self.gt_dir) if is_valid_file(f)}

        # Match corresponding pairs by identical filename to keep pairs strictly aligned
        common_filenames = sorted(list(noisy_files.intersection(gt_files)))

        if len(common_filenames) == 0:
            raise FileNotFoundError(
                f"No matching .npy file pairs found between '{self.noisy_dir}' and '{self.gt_dir}'."
            )

        self.filenames = common_filenames
        print(f"KLADataset: Found {len(self.filenames)} paired NoisyLR/GT samples.")

    def __len__(self) -> int:
        # Returns total number of paired samples
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads paired sample at index `idx`.

        Returns:
            noisy_lr (torch.Tensor): Input low-resolution noisy tensor of shape [1, 128, 128], torch.float32
            gt (torch.Tensor): Target high-resolution ground truth tensor of shape [1, 256, 256], torch.float32
        """
        filename = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        # 1. Load original numpy arrays from disk (Read-only operation; .npy files are untouched)
        noisy_arr = np.load(noisy_path)  # Input shape: (128, 128)
        gt_arr = np.load(gt_path)        # Target shape: (256, 256)

        # 2. Convert NumPy arrays to torch.float32 tensors
        noisy_tensor = torch.from_numpy(noisy_arr).float()
        gt_tensor = torch.from_numpy(gt_arr).float()

        # 3. Add channel dimension [1, H, W]
        if noisy_tensor.ndim == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)  # [128, 128] -> [1, 128, 128]
        elif noisy_tensor.ndim == 3 and noisy_tensor.shape[0] != 1:
            noisy_tensor = noisy_tensor.permute(2, 0, 1)

        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor.unsqueeze(0)        # [256, 256] -> [1, 256, 256]
        elif gt_tensor.ndim == 3 and gt_tensor.shape[0] != 1:
            gt_tensor = gt_tensor.permute(2, 0, 1)

        # 4. Consistently normalize GT target to [0, 1] range
        if self.normalize_gt:
            gt_tensor = torch.clamp(gt_tensor, 0.0, 1.0)

        # 5. Preserving NoisyLR noise information (No blind clipping applied to noisy_tensor)

        return noisy_tensor, gt_tensor


def create_train_val_datasets(
    root_dir: str = "dataset/KLA/train/train",
    val_ratio: float = 0.20,
    seed: int = 42,
    normalize_gt: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Creates training and validation datasets using a deterministic 80/20 split (seed=42).

    Args:
        root_dir (str): Path to root folder containing 'NoisyLR' and 'GT' subfolders.
        val_ratio (float): Fraction of data for validation (default: 0.20 = 20%).
        seed (int): Fixed random seed for deterministic reproducibility (default: 42).
        normalize_gt (bool): If True, normalizes GT target tensors to [0.0, 1.0].

    Returns:
        train_dataset (Dataset): Training dataset subset (80% of total samples).
        val_dataset (Dataset): Validation dataset subset (20% of total samples).
    """
    full_dataset = KLADataset(root_dir=root_dir, normalize_gt=normalize_gt)

    total_samples = len(full_dataset)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size

    # Use fixed PyTorch random generator seed for deterministic 80/20 split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Deterministic Split (seed={seed}, split={int((1-val_ratio)*100)}/{int(val_ratio*100)}):")
    print(f"  Total Samples:      {total_samples}")
    print(f"  Training Subset:    {len(train_dataset)} samples")
    print(f"  Validation Subset:  {len(val_dataset)} samples")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test section: Load one sample and print input/target shapes and min/max values
    print("=" * 60)
    print("                KLA DATASET SELF-TEST                  ")
    print("=" * 60)

    train_dataset, val_dataset = create_train_val_datasets(
        root_dir="dataset/KLA/train/train",
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )

    # Fetch first sample pair from training subset
    input_tensor, target_tensor = train_dataset[0]

    print("\n--- Loaded Sample Statistics ---")
    print(f"Input Shape (NoisyLR):  {list(input_tensor.shape)}")
    print(f"Target Shape (GT):      {list(target_tensor.shape)}")
    print(f"Input Min / Max:        {input_tensor.min().item():.6f} / {input_tensor.max().item():.6f}")
    print(f"Target Min / Max:       {target_tensor.min().item():.6f} / {target_tensor.max().item():.6f}")
    print("=" * 60)
