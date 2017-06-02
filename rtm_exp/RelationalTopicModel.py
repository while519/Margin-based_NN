import pickle
import logging

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

from ptm import RelationalTopicModel
from ptm.utils import convert_cnt_to_list, get_top_words

logger = logging.getLogger('RelationalTopicModel')
logger.propagate=False

if __name__ == '__main__':
    doc_ids = pickle.load(open('./data/cora/doc_ids.pkl', 'rb'))
    doc_cnt = pickle.load(open('./data/cora/doc_cnt.pkl', 'rb'))
    doc_symlinks = pickle.load(open('./data/cora/doc_links_sym.pkl', 'rb'))
    voca = pickle.load(open('./data/cora/voca.pkl', 'rb'))
    doc_asymlinks = pickle.load(open('./data/cora/doc_links_sym.pkl', 'rb'))

    n_doc = len(doc_ids)
    n_topic = 10
    n_voca = len(voca)
    max_iter = 50

    np.random.seed(213)

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

        # for symmetric links
        filtered_symlinks = [i for i in doc_symlinks[di] if i in trids]
        tr_doc_symlinks[di] = filtered_symlinks

    for di in teids:                 # testing citation pairs
        teIdxr += doc_asymlinks[di]
        teIdxl += [di] * len(doc_asymlinks[di])

        # symmetric links to feed RTM
        tr_doc_symlinks[di] = []



    model = RelationalTopicModel(n_topic, n_doc, n_voca, verbose=True)
    model.fit(doc_ids, doc_cnt, tr_doc_symlinks, max_iter=max_iter)

    for k in range(n_topic):
        top_words = get_top_words(model.beta, voca, k, 10)
        print('Topic', k, ':', ','.join(top_words))

