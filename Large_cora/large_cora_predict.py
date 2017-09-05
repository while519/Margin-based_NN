#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modelx import *
from scipy.io.matlab import loadmat
from scipy.io.matlab import savemat
import logging
import math
import time
import pickle
import scipy.sparse as sp

# experimental parameters
dataname = 'cora'
applyfn = 'softcauchy'

# adjustable parameters
outdim = 20
marge_ratio = 1.
reg = 1.

FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
_log = logging.getLogger(dataname +' experiment')
_log.setLevel(logging.DEBUG)
ch_file = logging.FileHandler(filename= dataname + '_lemb_' + applyfn + str(outdim)
                                        + '_margeratio' + str(marge_ratio) + '_reg' + str(reg) + '.log', mode='w')
ch_file.setLevel(logging.DEBUG)
ch_file.setFormatter(FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(FORMAT)
ch.setLevel(logging.DEBUG)
_log.addHandler(ch)
_log.addHandler(ch_file)


# ----------------------------------------------------------------------------
def SGDexp(state):
    _log.info(state)
    np.random.seed(state.seed)

    np.random.seed(213)
    # load the matlab data file
    doc_ids = pickle.load(open('./doc_ids.pkl', 'rb'))
    doc_symlinks = pickle.load(open('./doc_links_sym.pkl', 'rb'))
    doc_asymlinks = pickle.load(open('./doc_links_asym.pkl', 'rb'))

    n_doc = len(doc_ids)

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

    # split the data into training/testing set
    state.n_train = n_train
    state.n_test = n_test

    state.trIdxl = trIdxl
    state.trIdxr = trIdxr
    state.teIdxl = teIdxl
    state.teIdxr = teIdxr


    state.ntrain = len(state.trIdxl)
    state.ntest = len(state.teIdxl)
    state.train = np.mean(RankScoreIdx(simi_X, state.trIdxl, state.trIdxr))
    _log.debug('Content Only: Training set Mean Rank: %s ' % (state.train,))
    state.test = np.mean(RankScoreIdx(simi_X, state.teIdxl, state.teIdxr))
    _log.debug('Content Only: Testing set Mean Rank: %s ' % (state.test,))


    # initialize
    embedding = Embeddings(np.random, state.nsamples, state.outdim)  # N x K

    # Function compilation
    apply_fn = eval(state.applyfn)
    trainfunc = trainFn2Member(apply_fn, embedding, state.Q, state.marge, state.reg)

    out = []
    outb = []
    outc = []
    outd = []
    batchsize = math.floor(state.ntrain / state.nbatches)
    state.bestout = np.inf

    _log.info('BEGIN TRAINING')
    timeref = time.time()
    for epoch_count in range(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(state.ntrain)
        trainIdxl = state.trIdxl[order]
        trainIdxr = state.trIdxr[order]

        listidx = np.arange(state.nsamples, dtype='int32')
        listidx = listidx[np.random.permutation(len(listidx))]
        trainIdxrn = listidx[np.arange(state.ntrain) % len(listidx)]


        for _ in range(10):
            for ii in range(state.nbatches):
                tmpl = trainIdxl[ii * batchsize: (ii + 1) * batchsize]
                tmpr = trainIdxr[ii * batchsize: (ii + 1) * batchsize]
                tmprn = trainIdxrn[ii * batchsize: (ii + 1) * batchsize]
                outtmp = trainfunc(tmpl, tmpr, tmprn, state.lrmapping)
                out += [outtmp[0]]
                outb += [outtmp[1]]
                outc += [outtmp[2]]
                outd += [outtmp[3]]
                # mapping.normalize()

            if np.mean(out) <= state.bestout:
                state.bestout = np.mean(out)
                state.lrmapping *= 1.01
            else:
                state.lrmapping *= .4

        if (epoch_count % state.neval) == 0:
            _log.info('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref) / state.neval))
            _log.info('Cost mean: %s +/- %s      updates: %s%% ' % (np.mean(out), np.std(out), np.mean(outb) * 100))
            _log.debug('Learning rate: %s LeaveOneOut: %s  Entropy: %s' %
                       (state.lrmapping, np.mean(outc), np.mean(outd)))

            timeref = time.time()
            Dist = L2distance(embedding.E)
            Pr = apply_fn(Dist).eval()
            state.train = np.mean(RankScoreIdx(Pr, state.trIdxl, state.trIdxr))
            _log.debug('Training set Mean Rank: %s  Score: %s' % (state.train, np.mean(Pr[state.trIdxr, state.trIdxl])))
            state.test = np.mean(RankScoreIdx(Pr, state.teIdxl, state.teIdxr))
            _log.debug('Testing set Mean Rank: %s  Score: %s' % (state.test, np.mean(Pr[state.teIdxr, state.teIdxl])))
            state.cepoch = epoch_count
            savemat(dataname + '_emb_dim' + str(state.outdim) + '_method' + state.applyfn +
                    '_marge' + str(state.marge_ratio) + '_reg' + str(reg) +  '.mat', {'mappedX': embedding.E.eval()})
            _log.debug('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()

        outb = []
        outc = []
        out = []
        outd = []
        state.bestout = np.inf
        if state.lrmapping < state.baselr or (epoch_count // 2000):      # if the learning rate is not growing
            state.baselr *= 0.4
        state.lrmapping = state.baselr
        # f = open(state.savepath + '/' + 'state.pkl', 'wb')
        # pickle.dump(state, f, -1)
        # f.close()


if __name__ == '__main__':
    _log.info('Start')
    state = DD()

    # check the datapath
    datapath = '../New_data/'
    assert datapath is not None

    if 'Output' not in os.listdir('../'):
        os.mkdir('../Output')
    state.savepath = '../Output'

    # load the matlab data file
    doc_ids = pickle.load(open(datapath + dataname + '/doc_ids.pkl', 'rb'))
    doc_cnt = pickle.load(open(datapath + dataname + '/doc_cnt.pkl', 'rb'))
    doc_symlinks = pickle.load(open(datapath + dataname + '/doc_links_sym.pkl', 'rb'))
    voca = pickle.load(open(datapath + dataname + '/voca.pkl', 'rb'))
    doc_asymlinks = pickle.load(open(datapath + dataname + '/doc_links_asym.pkl', 'rb'))

    n_doc = len(doc_cnt)
    n_voca = len(voca)
    X = sp.lil_matrix((n_doc, n_voca))
    for di in range(n_doc):
        for idx, wi in enumerate(doc_ids[di]):
            X[di, wi] = doc_cnt[di][idx]

    preprocessing_tennique = 'ca'

    if preprocessing_tennique == 'lsa':
        u, s, vt = sp.linalg.svds(X, 300, which='LM')
        X = u
    elif preprocessing_tennique == 'ca':
        u, s, vt = sp.linalg.svds(X, 300, which='LM')
        X = u * np.sqrt(s)

    state.link_list = doc_asymlinks
    state.seed = 213
    state.totepochs = 500
    state.lrmapping = 100.
    state.baselr = state.lrmapping
    state.nsamples, state.nfeatures = np.shape(X)
    state.outdim = outdim
    state.applyfn = applyfn
    state.marge_ratio = marge_ratio
    state.marge = marge_ratio / state.nsamples
    state.nbatches = 100  # mini-batch SGD is not helping here
    state.neval = 10
    state.initial_dim = 300
    state.reg = reg
    state.perplexity = 20



    # cosine similarity measure
    simi_X = consine_simi(X)
    np.fill_diagonal(simi_X, 0)

    #Y = pca(X, state.initial_dim)
    # Compute P-values
    Q = x2p(X, 1e-5, state.perplexity)
    Q = np.maximum(Q, 1e-12)
    np.fill_diagonal(Q, 0)
    _log.info('Maximum probability value of the fixed perplexitied distribution: %s' % (np.max(Q, axis=None),))
    state.Q = T.as_tensor_variable(np.asarray(Q.T, dtype=theano.config.floatX))

    # start the experiments
    SGDexp(state)