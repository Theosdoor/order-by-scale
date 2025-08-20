import torch
from torch.utils.data import TensorDataset
import itertools

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
torch.manual_seed(0)

def get_dataset(
    list_len=2, # [d1, d2]
    n_digits=100,
    train_split=0.8, # 80% train, 20% test
    train_dupes_only=False, # whether to remove duplicates (where d1 == d2) from the validation set
    no_dupes=False, # whether to use only non-duplicates (i.e. all d1 != d2)
    mask_tok=None, # special masking token for o1 and o2
    sep_tok=None, # special seperator token for the model to think about the input 
):
    seq_len = list_len * 2 + 1 # [d1, d2, SEP, o1, o2]
    digits = list(range(n_digits)) # 100 digits from 0 to 99
    if mask_tok is None:
        mask_tok = n_digits 
    if sep_tok is None:
        sep_tok = n_digits + 1 

    # Create all possible combinations of digits
    all_data = list(itertools.product(digits, repeat=list_len))
    all_data = torch.tensor(all_data, dtype=torch.int64)

    # Split into dupes (d1 == d2) and non-dupes
    dupes_mask = all_data[:, 0] == all_data[:, 1]
    dupes = all_data[dupes_mask]
    non_dupes = all_data[~dupes_mask]

    def build_inputs_targets(data_tensor: torch.Tensor):
        n = len(data_tensor)
        targets = torch.full((n, seq_len), sep_tok, dtype=torch.int64)
        targets[:, :list_len] = data_tensor
        targets[:, list_len + 1 :] = data_tensor
        inputs = targets.clone()
        inputs[:, list_len + 1 :] = mask_tok
        return inputs, targets

    if no_dupes:
        # Use only non-duplicates for both train and val
        all_inputs, all_targets = build_inputs_targets(non_dupes)
        n_data = len(non_dupes)

        # Shuffle together
        perm = torch.randperm(n_data)
        all_inputs = all_inputs[perm]
        all_targets = all_targets[perm]

        # Split into train/val
        split = int(train_split * n_data)
        train_inputs = all_inputs[:split]
        train_targets = all_targets[:split]
        val_inputs = all_inputs[split:]
        val_targets = all_targets[split:]

    else: # if allowed dupes
        if train_dupes_only:
            # Split non-dupes into train/val, then add dupes only to train and reshuffle
            nd_inputs, nd_targets = build_inputs_targets(non_dupes)
            n_nd = len(non_dupes)

            # Shuffle non-dupes together
            perm = torch.randperm(n_nd)
            nd_inputs = nd_inputs[perm]
            nd_targets = nd_targets[perm]

            # Split non-dupes into train/val
            split = int(train_split * n_nd)
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
            split = int(train_split * n_data)
            train_inputs = all_inputs[:split]
            train_targets = all_targets[:split]
            val_inputs = all_inputs[split:]
            val_targets = all_targets[split:]

    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds = TensorDataset(val_inputs, val_targets)

    return train_ds, val_ds
