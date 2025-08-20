# %%
import torch
from torch.utils.data import TensorDataset

import os
import itertools

# %%
LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]
TRAIN_DUPES_ONLY = True # whether to add duplicates to training set only (for dupes dataset)
NO_DUPES = False # whether to use the no-dupes test dataset (i.e. d1 != d2)

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
PAD = N_DIGITS # special padding token
SEP = N_DIGITS + 1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)

TRAIN_SPLIT = 0.8 # 80% train, 20% test

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
device = DEV
torch.manual_seed(0)

# %%
def get_dataset():
    # Create all possible combinations of digits
    all_data = list(itertools.product(DIGITS, repeat=LIST_LEN))
    all_data = torch.tensor(all_data, dtype=torch.int64)

    # Split into dupes (d1 == d2) and non-dupes
    dupes_mask = all_data[:, 0] == all_data[:, 1]
    dupes = all_data[dupes_mask]
    non_dupes = all_data[~dupes_mask]

    def build_inputs_targets(data_tensor: torch.Tensor):
        n = len(data_tensor)
        targets = torch.full((n, SEQ_LEN), SEP, dtype=torch.int64)
        targets[:, :LIST_LEN] = data_tensor
        targets[:, LIST_LEN + 1 :] = data_tensor
        inputs = targets.clone()
        inputs[:, LIST_LEN + 1 :] = PAD
        return inputs, targets

    if NO_DUPES:
        # Use only non-duplicates for both train and val
        all_inputs, all_targets = build_inputs_targets(non_dupes)
        n_data = len(non_dupes)

        # Shuffle together
        perm = torch.randperm(n_data)
        all_inputs = all_inputs[perm]
        all_targets = all_targets[perm]

        # Split into train/val
        split = int(TRAIN_SPLIT * n_data)
        train_inputs = all_inputs[:split]
        train_targets = all_targets[:split]
        val_inputs = all_inputs[split:]
        val_targets = all_targets[split:]

    else: # if allowed dupes
        if TRAIN_DUPES_ONLY:
            # Split non-dupes into train/val, then add dupes only to train and reshuffle
            nd_inputs, nd_targets = build_inputs_targets(non_dupes)
            n_nd = len(non_dupes)

            # Shuffle non-dupes together
            perm = torch.randperm(n_nd)
            nd_inputs = nd_inputs[perm]
            nd_targets = nd_targets[perm]

            # Split non-dupes into train/val
            split = int(TRAIN_SPLIT * n_nd)
            train_inputs = nd_inputs[:split]
            train_targets = nd_targets[:split]
            val_inputs = nd_inputs[split:]
            val_targets = nd_targets[split:]

            # Build dupes and append to train only
            d_inputs, d_targets = build_inputs_targets(dupes)
            train_inputs = torch.cat([train_inputs, d_inputs], dim=0)
            train_targets = torch.cat([train_targets, d_targets], dim=0)

            # Shuffle the augmented training set
            perm_train = torch.randperm(len(train_inputs))
            train_inputs = train_inputs[perm_train]
            train_targets = train_targets[perm_train]
        else:
            # Use all data (dupes + non-dupes) for both train and val
            all_inputs, all_targets = build_inputs_targets(all_data)
            n_data = len(all_data)

            # Shuffle together
            perm = torch.randperm(n_data)
            all_inputs = all_inputs[perm]
            all_targets = all_targets[perm]

            # Split into train/val
            split = int(TRAIN_SPLIT * n_data)
            train_inputs = all_inputs[:split]
            train_targets = all_targets[:split]
            val_inputs = all_inputs[split:]
            val_targets = all_targets[split:]

    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds = TensorDataset(val_inputs, val_targets)

    return train_ds, val_ds

# %%
if __name__ == "__main__":
    dupes_suffix = 'nodupes' if NO_DUPES else 'dupes'
    if TRAIN_DUPES_ONLY and not NO_DUPES:
        dupes_suffix += '_traindupesonly'

    DATASET_NAME = f"listlen{LIST_LEN}_digits{N_DIGITS}_{dupes_suffix}"
    DATASET_PATH = f"data/{DATASET_NAME}.pt"

    if os.path.exists(DATASET_PATH):
        print(f"{DATASET_PATH} already exists. Please delete it or change the dataset name.")
        raise FileExistsError()

    train_ds, val_ds = get_dataset()

    torch.save({
        'train': train_ds,
        'val': val_ds
    }, DATASET_PATH)

    print(f"Dataset saved to {DATASET_PATH}")
    print("Train dataset size:", len(train_ds))
    print("Validation dataset size:", len(val_ds))
    print("Input example:", train_ds[0][0])
    print("Target example:", train_ds[0][1])


