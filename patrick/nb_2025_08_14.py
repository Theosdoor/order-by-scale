# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("results.csv")


# %%

if __name__ == "__main__":
    standard_df = df[
        (df["N_DIGITS"] == 100)
        & (df["D_MODEL"] == 128)
        & (df["N_HEAD"] == 1)
        & (df["USE_LN"] == False)
        & (df["USE_BIAS"] == False)
        & (df["FREEZE_WV"] == True)
        & (df["FREEZE_WO"] == True)
        & (df["WEIGHT_DECAY"] == 0.01)
        & (df["LIST_LEN"] <= 10)
        & (df["N_LAYERS"] <= 10)
        & (1 < df["N_LAYERS"])
    ].copy()
    standard_df = standard_df[["LIST_LEN", "N_LAYERS", "run_idx", "val_acc"]]
    standard_df = standard_df.groupby(
        ["LIST_LEN", "N_LAYERS"], as_index=False
    ).val_acc.mean()


    # pivot to a grid
    heat = (standard_df
            .pivot_table(index='LIST_LEN', columns='N_LAYERS', values='val_acc', aggfunc='mean')
            .sort_index().sort_index(axis=1))

    fig, ax = plt.subplots()
    im = ax.imshow(heat.values, origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Validation accuracy')

    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticklabels(heat.columns)
    ax.set_yticklabels(heat.index)
    ax.set_xlabel('N_LAYERS')
    ax.set_ylabel('LIST_LEN')
    ax.set_title('Validation performance')

    # label each cell
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if np.isnan(v): 
                continue
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()


# %%
