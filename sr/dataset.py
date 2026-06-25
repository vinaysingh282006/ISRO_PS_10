import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class TIRSuperResolutionDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        for scene in os.listdir(root_dir):

            scene_path = os.path.join(root_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            for sample in os.listdir(scene_path):

                sample_path = os.path.join(
                    scene_path,
                    sample
                )

                lr_file = os.path.join(
                    sample_path,
                    "tir_200m.npy"
                )

                hr_file = os.path.join(
                    sample_path,
                    "tir_100m_512.npy"
                )

                if (
                    os.path.exists(lr_file)
                    and
                    os.path.exists(hr_file)
                ):
                    self.samples.append(
                        (lr_file, hr_file)
                    )

        print(
            f"Found {len(self.samples)} SR samples"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        lr_path, hr_path = self.samples[idx]

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        # ----------------------------------
        # NORMALIZATION
        # ----------------------------------

        lr = (lr - lr.min()) / (lr.max() - lr.min() + 1e-8)

        hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-8)

        # ----------------------------------
        # DATA AUGMENTATION
        # ----------------------------------

        if random.random() < 0.5:

            lr = np.flip(
                lr,
                axis=2
            ).copy()

            hr = np.flip(
                hr,
                axis=2
            ).copy()

        if random.random() < 0.5:

            lr = np.flip(
                lr,
                axis=1
            ).copy()

            hr = np.flip(
                hr,
                axis=1
            ).copy()

        k = random.randint(0, 3)

        lr = np.rot90(
            lr,
            k,
            axes=(1, 2)
        ).copy()

        hr = np.rot90(
            hr,
            k,
            axes=(1, 2)
        ).copy()

        # ----------------------------------
        # NUMPY -> TORCH
        # ----------------------------------

        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return lr, hr


if __name__ == "__main__":

    dataset = TIRSuperResolutionDataset(
        r"C:\Users\Vinay Singh\Desktop\ISRO_PS_10\output\patches"
    )

    print("\nDataset Size:", len(dataset))

    lr, hr = dataset[0]

    print("LR Shape:", lr.shape)
    print("HR Shape:", hr.shape)

    print("LR Min:", lr.min().item())
    print("LR Max:", lr.max().item())

    print("HR Min:", hr.min().item())
    print("HR Max:", hr.max().item())





