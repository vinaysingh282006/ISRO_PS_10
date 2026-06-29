import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset


class RGBDataset(Dataset):

    def __init__(

        self,

        root_dir,

        patch_size=256,

        training=True,

        samples=None

    ):

        self.root_dir = root_dir

        self.patch_size = patch_size

        self.training = training

        self.samples = []

        # ---------------------------------------------
        # PREDEFINED TRAIN / VALIDATION SPLIT
        # ---------------------------------------------

        if samples is not None:

            self.samples = samples

            print(f"Loaded {len(self.samples)} predefined samples")

            return

        print()

        print("=" * 60)

        print("Scanning Dataset...")

        print("=" * 60)

        skipped = 0

        # ---------------------------------------------
        # SCAN DATASET
        # ---------------------------------------------

        for folder in sorted(os.listdir(root_dir)):

            folder_path = os.path.join(

                root_dir,

                folder

            )

            if not os.path.isdir(folder_path):

                continue

            for sample in sorted(os.listdir(folder_path)):

                sample_path = os.path.join(

                    folder_path,

                    sample

                )

                if not os.path.isdir(sample_path):

                    continue

                tir = os.path.join(

                    sample_path,

                    "tir_100m_512.npy"

                )

                rgb = os.path.join(

                    sample_path,

                    "rgb_100m_512.npy"

                )

                if not (

                    os.path.exists(tir)

                    and

                    os.path.exists(rgb)

                ):

                    skipped += 1

                    continue

                try:

                    tir_test = np.load(

                        tir,

                        mmap_mode="r"

                    )

                    rgb_test = np.load(

                        rgb,

                        mmap_mode="r"

                    )

                    if tir_test.shape != (1, 512, 512):

                        raise ValueError

                    if rgb_test.shape != (3, 512, 512):

                        raise ValueError

                    self.samples.append(

                        (

                            tir,

                            rgb

                        )

                    )

                except Exception:

                    skipped += 1

                    continue

        print()

        print(f"Valid Samples   : {len(self.samples)}")

        print(f"Skipped Samples : {skipped}")

        print("=" * 60)

        # =====================================================
    # DATASET SIZE
    # =====================================================

    def __len__(self):

        return len(self.samples)

    # =====================================================
    # LOAD SAMPLE
    # =====================================================

    def __getitem__(self, idx):

        while True:

            tir_path, rgb_path = self.samples[idx]

            try:

                tir = np.load(tir_path).astype(np.float32)
                rgb = np.load(rgb_path).astype(np.float32)

                if tir.shape != (1, 512, 512):
                    raise ValueError("Invalid thermal shape")

                if rgb.shape != (3, 512, 512):
                    raise ValueError("Invalid RGB shape")

                break

            except Exception:

                # Pick another random sample if this one fails
                idx = random.randint(0, len(self.samples) - 1)

        # -------------------------------------------------
        # NORMALIZE
        # -------------------------------------------------

        tir /= 65535.0
        rgb /= 65535.0

        # -------------------------------------------------
        # RANDOM / CENTER CROP
        # -------------------------------------------------

        _, H, W = tir.shape

        ps = self.patch_size

        if self.training:

            x = random.randint(0, W - ps)
            y = random.randint(0, H - ps)

        else:

            x = (W - ps) // 2
            y = (H - ps) // 2

        tir = tir[:, y:y+ps, x:x+ps]
        rgb = rgb[:, y:y+ps, x:x+ps]


        if self.training:

            if random.random() < 0.5:
                tir = np.flip(tir, axis=2).copy()
                rgb = np.flip(rgb, axis=2).copy()

            if random.random() < 0.5:
                tir = np.flip(tir, axis=1).copy()
                rgb = np.flip(rgb, axis=1).copy()

                k = random.randint(0, 3)
                tir = np.rot90(tir, k, axes=(1, 2)).copy()
                rgb = np.rot90(rgb, k, axes=(1, 2)).copy()


        # -------------------------------------------------
        # CONTIGUOUS MEMORY
        # -------------------------------------------------

        tir = np.ascontiguousarray(tir)
        rgb = np.ascontiguousarray(rgb)

        return (

            torch.from_numpy(tir),

            torch.from_numpy(rgb)

        )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    DATASET_PATH = r"C:\Users\Vinay Singh\Desktop\OUTPUT_ARCHIVED\patches"

    dataset = RGBDataset(

        DATASET_PATH,

        patch_size=256,

        training=True

    )

    print()

    print("=" * 60)

    print("Dataset Size :", len(dataset))

    print("=" * 60)

    tir, rgb = dataset[0]

    print()

    print("Thermal Shape :", tir.shape)
    print("RGB Shape     :", rgb.shape)

    print()

    print("Thermal Range :", tir.min().item(), "->", tir.max().item())
    print("RGB Range     :", rgb.min().item(), "->", rgb.max().item())

    print()

    print("=" * 60)
    print("Dataset Loader Working Successfully")
    print("=" * 60)
