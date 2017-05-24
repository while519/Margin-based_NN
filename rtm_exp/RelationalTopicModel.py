import pickle
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ptm import RelationalTopicModel
from ptm.utils import convert_cnt_to_list, get_top_words

logger = logging.getLogger('RelationalTopicModel')
logger.propagate=False

if __name__ == '__main__':
    doc_ids = pickle.load(open('./data/cora/doc_ids.pkl', 'rb'))
    doc_cnt = pickle.load(open('./data/cora/doc_cnt.pkl', 'rb'))
    doc_links = pickle.load(open('./data/cora/doc_links_sym.pkl', 'rb'))
    voca = pickle.load(open('./data/cora/voca.pkl', 'rb'))

    n_doc = len(doc_ids)
    n_topic = 10
    n_voca = len(voca)
    max_iter = 50

    model = RelationalTopicModel(n_topic, n_doc, n_voca, verbose=True)
    model.fit(doc_ids, doc_cnt, doc_links, max_iter=max_iter)

    for k in range(n_topic):
        top_words = get_top_words(model.beta, voca, k, 10)
        print('Topic', k, ':', ','.join(top_words))