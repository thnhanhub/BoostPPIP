"""
Performance on Yeast datasets:
1. Using 5-fold cross-validation on "Yeast" datasets.
2. Params selection

@author: thnhan
"""
import os
import pickle
import time

import numpy as np

from models.boostppip_model import net_embedding
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from datasets.dset_tool import load_raw_dset
from feature_extraction.prepare import prot_to_token

from utils.report_result import print_metrics, my_cv_report


def get_avelen(inds, dset):
    pos_A, pos_B, neg_A, neg_B = dset['seq_pairs']
    pos_AB = np.hstack((pos_A, pos_B))
    neg_AB = np.hstack((neg_A, neg_B))
    prots = np.concatenate((pos_AB, neg_AB), axis=0)
    prots = prots.flatten()
    prots = prots[inds]
    prots = np.unique(prots)
    do_dai = [len(seq) for seq in prots]
    avelen = int(sum(do_dai) / len(do_dai))
    return avelen


def prepare_Yeast_token(Vocal_W1, protlen, dset):
    pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']

    # --- Lấy đặc trưng bằng Word2vec
    seqs_A = np.concatenate([pos_seq_A, neg_seq_A], axis=0)
    seqs_B = np.concatenate([pos_seq_B, neg_seq_B], axis=0)
    tokens_A = prot_to_token(Vocal_W1, seqs_A, protlen)
    tokens_B = prot_to_token(Vocal_W1, seqs_B, protlen)

    return tokens_A, tokens_B


def eval_model(pairs):
    start_time = time.time()

    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_Y, cv_test_y = [], []

    method_result = dict()
    for i, (tr_inds, te_inds) in enumerate(skf.split(pairs, labels)):
        print("\nFold", i)
        protlen = get_avelen(tr_inds, yeast_dset)
        print("Average length:", protlen)

        tokens_A, tokens_B = prepare_Yeast_token(Vocal_W1, protlen, yeast_dset)

        tr_A_w2v, = tokens_A[tr_inds],
        tr_B_w2v, = tokens_B[tr_inds],

        te_A_w2v, = tokens_A[te_inds],
        te_B_w2v, = tokens_B[te_inds],

        Y = to_categorical(labels)
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        name = 'results/5BoostPPIP_trained_on_yeast_fold'
        # --- DEF MODEL
        if os.path.exists(name + str(i) + '.h5'):
            model = load_model(name + str(i) + '.h5')
        else:
            model = net_embedding(protlen, embedding=Vocal_W1, n_units=1024)

            opt = Adam(decay=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # --- FIT MODEL
            hist = model.fit([tr_A_w2v, tr_B_w2v], tr_Y,
                             epochs=50,
                             batch_size=32,  # 32
                             verbose=0)
            hists.append(hist)

            # --- SAVE MODEL
            model.save(name + str(i) + ".h5")

        # --- REPORT
        prob_Y = model.predict([te_A_w2v, te_B_w2v])

        # --- Keep for comparing with methods, thnhan
        method_result['fold' + str(i)] = {"true_y": np.argmax(te_Y, axis=1),
                                          "prob_Y": prob_Y}
        pickle.dump(method_result, open(name + '.pkl', 'wb'))
        # ---

        scr = print_metrics(np.argmax(te_Y, axis=1), prob_Y)
        scores.append(scr)

        cv_prob_Y.append(prob_Y[:, 1])
        cv_test_y.append(np.argmax(te_Y, axis=1))

    # --- FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    print("Running time", time.time() - start_time)

    with open("results/results_yeast.txt", "a") as fout:
        fout.write("#\n")
        for scr in scores_array:
            fout.write(",".join([a[0] + ":" + str(a[1]) for a in scr.items()]) + "\n")

    # --- Plot AUC, AUPR
    plot_folds(plt, cv_test_y, cv_prob_Y)
    plt.show()

    return hists


if __name__ == "__main__":
    # --- GLOBAL HYPER PARAMETERS
    Vocal_W1 = pickle.load(open("feature_extraction/trained_W1_20.pkl", "rb"))
    # ------

    yeast_dset, summary = load_raw_dset("datasets/Yeast")
    id_pairs = yeast_dset['id_pairs']
    labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))
    eval_model(id_pairs)
