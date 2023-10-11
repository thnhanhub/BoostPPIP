# -*- coding: utf-8 -*-
"""
@author: thnhan
"""
import os.path
import pickle
from sys import path
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from datasets.dset_tool import load_raw_dset
from datasets.fasta_tool import get_protein_from_fasta
from feature_extraction.prepare import prot_to_token
from models.boostppip_model import net_embedding
from experiment_on_yeast import prepare_Yeast_token, get_avelen


def load_trained_model():
    trained_model = load_model(model_filename)
    print('--- Loaded trained model')
    return trained_model


def prepare_testset(Vocal_W1, protfile_A, protfile_B, fixlen):
    protseq_A = get_protein_from_fasta(protfile_A)
    protseq_B = get_protein_from_fasta(protfile_B)
    tokens_A = prot_to_token(Vocal_W1, protseq_A, fixlen)
    tokens_B = prot_to_token(Vocal_W1, protseq_B, fixlen)
    labels = np.ones(len(protseq_A), dtype=int)  # labels
    return tokens_A, tokens_B, labels


def run_test(file_prot_A, file_prot_B, dataset_name):
    tokens_A, tokens_B, te_labels = prepare_testset(Vocal_W1,
                                                    file_prot_A,
                                                    file_prot_B,
                                                    protlen)  # label = 1
    # print(tokens_A)
    y_prob = tr_model.predict([tokens_A, tokens_B])

    # --- SAVE predictions
    pickle.dump(y_prob, open('predictions_on_' + dataset_name, 'wb'))
    # ---

    y_pred = np.argmax(y_prob, axis=1).astype(int)
    ACC = sum(y_pred == te_labels) / len(y_pred)
    print('> Accuracy: {:.2%}'.format(ACC))
    print('> Correct : {:d}/{:d}'.format(sum(y_pred == te_labels), len(te_labels)))
    return ACC


def on_species(dataset_name):
    print('\n--- Testing on {} ...'.format(dataset_name))
    file_prot_A = species_datadir + r'/' + dataset_name + r'_ProA.txt'
    file_prot_B = species_datadir + r'/' + dataset_name + r'_ProB.txt'
    return run_test(file_prot_A, file_prot_B, dataset_name)


def on_network_data(dataset_name):
    print('\n--- Testing on {} ...'.format(dataset_name))
    file_prot_A = path[0] + r'/' + network_datadir + r'/' + dataset_name + r'/' + dataset_name + r'_ProA.txt'
    file_prot_B = path[0] + r'/' + network_datadir + r'/' + dataset_name + r'/' + dataset_name + r'_ProB.txt'
    return run_test(file_prot_A, file_prot_B, dataset_name)


if __name__ == "__main__":
    # ====== GLOBAL HYPER PARAMETERS
    epochs = 64
    Vocal_W1 = pickle.load(open("feature_extraction/trained_W1_20.pkl", "rb"))
    species_datadir = r'datasets/Cross_species'
    network_datadir = r'datasets/Networks'
    loss_fn = 'categorical_crossentropy'

    model_filename = 'results/BoostPPIP_trained_on_FULL_Yeast_CBOW20.h5'
    results_filename = 'results/prediction_accuracies_on_independent_test.txt'

    # trained_w2v = Word2Vec.load(w2v_filename)
    yeast_dset, summary = load_raw_dset("datasets/Yeast")
    id_pairs = yeast_dset['id_pairs']
    tr_labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))

    inds = np.arange(len(id_pairs))
    protlen = get_avelen(inds, yeast_dset)
    print("Average length:", protlen)

    if os.path.exists(model_filename):
        tr_model = load_trained_model()
    else:
        print("\n--- Train model ...")

        tr_w2v_A, tr_w2v_B = prepare_Yeast_token(Vocal_W1, protlen, yeast_dset)
        tr_model = net_embedding(protlen, embedding=Vocal_W1, n_units=1024)

        opt = Adam(decay=0.001)
        tr_model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

        # --- FIT MODEL
        tr_labels = to_categorical(tr_labels)
        tr_model.fit([tr_w2v_A, tr_w2v_B], tr_labels,
                     batch_size=32,  # 64
                     epochs=50,  # 64,
                     verbose=1)

        tr_model.save(model_filename)
        print('\n--- SAVED {}.'.format(model_filename))

    # ====== TEST
    all_acc = dict()

    print('\n--- Test on PPIs network')
    all_acc.update({'Cancer_specific': on_network_data('Cancer_specific')})
    all_acc.update({'One_core': on_network_data('One_core')})
    all_acc.update({'Wnt_related': on_network_data('Wnt_related')})

    print('\n--- Test on cross-species')
    all_acc.update({'Celeg': on_species('Celeg')})
    all_acc.update({'Ecoli': on_species('Ecoli')})
    all_acc.update({'Hpylo': on_species('Hpylo')})
    all_acc.update({'Hsapi': on_species('Hsapi')})
    all_acc.update({'Mmusc': on_species('Mmusc')})

    # # ====== Lưu kết quả vào file
    # with open(results_filename, 'a') as f:
    #     f.write("#\n")
    #     for d, acc in all_acc.items():
    #         f.write(d + "\t" + str(acc) + "\n")
