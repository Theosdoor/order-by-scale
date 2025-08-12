# %%
import torch
from torch.utils.data import TensorDataset

import os
import itertools

# %%
LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]
NO_DUPES = True # whether to use the no-dupes test dataset (i.e. d1 != d2)

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
PAD = N_DIGITS # special padding token
SEP = N_DIGITS + 1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
TRAIN_SPLIT = 0.8 # 80% train, 20% test

# For backward compatibility with older versions
USE_PAD = True # whether to use the PAD token in the input sequences (or just SEP)

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
    n_data = len(all_data)
    all_data = torch.tensor(all_data, dtype=torch.int64)
    if NO_DUPES:
        # Filter out combinations where d1 == d2
        all_data = all_data[all_data[:, 0] != all_data[:, 1]]
        n_data = len(all_data)

    # Create sequences of the form [d1, d2, SEP, d1, d2]
    all_targets = torch.full((n_data, SEQ_LEN), SEP)
    all_targets[:, :LIST_LEN] = all_data
    all_targets[:, LIST_LEN+1:] = all_data

    # Create input sequences of the form [d1, d2, SEP, PAD, PAD]
    all_inputs = all_targets.clone()
    all_inputs[:, LIST_LEN+1:] = PAD if USE_PAD else SEP # backward compat

    # Shuffle the dataset (inputs and targets together)
    perm = torch.randperm(n_data)
    all_inputs = all_inputs[perm]
    all_targets = all_targets[perm]

    train_ds = TensorDataset(all_inputs[:int(TRAIN_SPLIT*n_data)], all_targets[:int(TRAIN_SPLIT*n_data)])  # 80% for training
    val_ds = TensorDataset(all_inputs[int(TRAIN_SPLIT*n_data):], all_targets[int(TRAIN_SPLIT*n_data):])  # 20% for validation
        
    return train_ds, val_ds

# %%
if __name__ == "__main__":
    DATASET_NAME = f"listlen{LIST_LEN}_digits{N_DIGITS}_{'nodupes' if NO_DUPES else 'dupes'}_{'usePAD' if USE_PAD else 'noPAD'}"
    DATASET_PATH = f"data/{DATASET_NAME}.pt"

    if os.path.exists(DATASET_PATH):
        raise FileExistsError(f"{DATASET_PATH} already exists. Please delete it or change the dataset name.")

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


