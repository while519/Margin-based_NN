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

# experimental parameters
dataname = 'citeseer'
applyfn = 'softcauchy'

# adjustable parameters
outdim = 20
marge_ratio = 20.
reg = 1.

FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
_log = logging.getLogger(dataname +' experiment')
_log.setLevel(logging.DEBUG)
ch_file = logging.FileHandler(filename= dataname + '_predictx_' + applyfn + str(outdim)
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

    # split the data into training/testing set
    state.ntrain = math.floor(.9 * state.nlinks)
    state.ntest = state.nlinks - state.ntrain
    indices = np.random.permutation(state.nlinks)
    state.trIdxl = state.Idxl[indices[: state.ntrain]]
    state.trIdxr = state.Idxr[indices[: state.ntrain]]

    state.teIdxl = state.Idxl[indices[state.ntrain :]]
    state.teIdxr = state.Idxr[indices[state.ntrain :]]

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


        for _ in range(20):
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
            savemat(dataname + '_predict_dim' + str(state.outdim) + '_method' + state.applyfn +
                    '_marge' + str(state.marge_ratio) + '_reg' + str(reg) + '.mat', {'mappedX': embedding.E.eval()})
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
        f = open(state.savepath + '/' + 'state.pkl', 'wb')
        pickle.dump(state, f, -1)
        f.close()


if __name__ == '__main__':
    _log.info('Start')
    state = DD()

    # check the datapath
    datapath = '../Data/'
    assert datapath is not None

    if 'Output' not in os.listdir('../'):
        os.mkdir('../Output')
    state.savepath = '../Output'

    # load the matlab data file
    mat = loadmat(datapath + dataname + '.mat')
    X = np.array(mat['X'], np.float32)
    I = np.array(mat['I'], np.float32)
    state.Idxl = np.asarray(I[:, 0].flatten() - 1, dtype='int32')  # numpy indexes start from 0
    state.Idxr = np.asarray(I[:, 1].flatten() - 1, dtype='int32')

    state.seed = 213
    state.totepochs = 2000
    state.lrmapping = 10.
    state.baselr = state.lrmapping
    state.nsamples, state.nfeatures = np.shape(X)
    state.nlinks = np.shape(state.Idxl)[0]
    state.outdim = outdim
    state.applyfn = applyfn
    state.marge_ratio = marge_ratio
    state.marge = marge_ratio / state.nsamples
    state.nbatches = 1  # mini-batch SGD is not helping here
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
