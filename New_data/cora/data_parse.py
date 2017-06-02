import pickle
import numpy as np
import math

if __name__ == '__main__':
    np.random.seed(213)
    # load the matlab data file
    doc_ids = pickle.load(open('./doc_ids.pkl', 'rb'))
    doc_cnt = pickle.load(open('./doc_cnt.pkl', 'rb'))
    doc_symlinks = pickle.load(open('./doc_links_sym.pkl', 'rb'))
    voca = pickle.load(open('./voca.pkl', 'rb'))
    doc_asymlinks = pickle.load(open('./doc_links_asym.pkl', 'rb'))

    n_doc = len(doc_ids)
    n_voca = len(voca)

    # splitting datasets
    n_train = math.floor(.9 * n_doc)
    n_test = n_doc - n_train
    indices = np.random.permutation(n_doc)
    trids = indices[: n_train]
    teids = indices[n_train :]

    Idxl = []
    Idxr = []
    tr_doc_symlinks = doc_symlinks
    for di in range(n_doc):                 # all citation pairs
        Idxl += [di] * len(doc_asymlinks[di])
        Idxr += doc_asymlinks[di]

    trIdxl = []
    trIdxr = []
    teIdxl = []
    teIdxr = []
    for di in trids:                 # training citation pairs
        tridxr = [i for i in doc_asymlinks[di] if i in trids]
        trIdxr += tridxr
        trIdxl += [di] * len(tridxr)
        teidxr = [i for i in doc_asymlinks[di] if i in teids]
        teIdxr += teidxr
        teIdxl += [di] * len(teidxr)
        filtered_symlinks = [i for i in doc_symlinks[di] if i in trids]
        tr_doc_symlinks[di] = filtered_symlinks

    for di in teids:                 # testing citation pairs
        teIdxr += doc_asymlinks[di]
        teIdxl += [di] * len(doc_asymlinks[di])
        tr_doc_symlinks[di] = []

    print('code terminated')
