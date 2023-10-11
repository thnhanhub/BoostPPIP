"""
@author: thnhan
"""

import pandas as pd
import numpy as np

from datasets.fasta_tool import fasta_to_dataframe


def load_raw_dset(dset_dir):
    seq = pd.read_csv(dset_dir + '/uniprotein.txt', index_col=0, sep="\t")
    pos = pd.read_csv(dset_dir + '/positive.txt', sep="\t")
    neg = pd.read_csv(dset_dir + '/negative.txt', sep="\t")

    do_dai = sorted([len(p) for p in seq.protein])
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    P_seq_A = seq.loc[pos.proteinA]['protein'].values
    P_seq_B = seq.loc[pos.proteinB]['protein'].values
    N_seq_A = seq.loc[neg.proteinA]['protein'].values
    N_seq_B = seq.loc[neg.proteinB]['protein'].values

    labels = np.array([1] * len(pos) + [0] * len(neg))
    pairs = np.vstack((pos.values, neg.values))
    dset = {"labels": labels, "id_pairs": pairs, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


if __name__ == "__main__":
    dset, summary = load_raw_dset(r"./Human8161")
    print("summary:", summary)
    print(dset.keys())
    P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq_pairs']
    print("pair\n", dset['pair'])
    print("label\n", dset['label'])
    print("P_seq_A\n", P_seq_A)
