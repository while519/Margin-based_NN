import os
from modelx import *
import numpy as np
import theano
import theano.tensor as T
from scipy.io.matlab import loadmat, savemat
import logging
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import pickle

def SGDexp(state, _log):
    np.random.seed(state.seed)
    # compute number of minibatches for training, validation and testing
    state.n_train_batches = state.n_samples // state.rbm_batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))


    # construct the RBM class
    lrbm = LayerRBM(input=state.train_set, n_visible=state.n_features,
              n_hidden=state.n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    if state.pretrain:
        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(np.zeros((state.rbm_batch_size, state.n_hidden),
                                                  dtype=theano.config.floatX),
                                         borrow=True)

        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = lrbm.get_cost_updates(lr=state.lr_rbm,
                                             persistent=persistent_chain, k=state.n_chains)

        #################################
        #     Training the RBM          #
        #################################
        if not os.path.isdir(state.savepath):
            os.makedirs(state.savepath)
        #os.chdir(state.savepath)

        # start-snippet-5
        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_lrbm = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                state.train_set: state.train_set[index * state.rbm_batch_size: (index + 1) * state.rbm_batch_size]
            },
            name='train_rbm'
        )


        plotting_time = 0.
        start_time = time.time()

        # go through training epochs
        for epoch in range(state.rbm_training_epochs):

            # go through the training set
            mean_cost = []
            for batch_index in range(state.n_train_batches):
                mean_cost += [train_lrbm(batch_index)]

            _log.info('Training epoch %d, cost is %s' % (epoch, np.mean(mean_cost)))

        end_time = time.time()

        pretraining_time = (end_time - start_time) - plotting_time

        _log.info('Training took %f s' % (pretraining_time, ))
        # end-snippet-5 start-snippet-6
    #################################
    #     updating from linkage     #
    #################################
    # find out the number of test samples
    # Function compilation
    if state.pretrain:
        if state.op == 'Mahalanobis':
            topLayer = LayerMahalanobis(rng, n_inp=state.n_hidden, n_out=state.n_out, tag=' Mahalanobis')
        elif state.op == 'Weight':
            topLayer = LayerWeight(rng, n_inp=state.n_hidden, tag=' Weight')
        else:
            topLayer = Layer()
        botLayer = lrbm

    elif not state.pretrain:
        botLayer = Layer(input=state.train_set)      # discarding pretraining step
        if state.op == 'Mahalanobis':
            topLayer = LayerMahalanobis(rng, n_inp=state.n_features, n_out=state.n_out, tag=' Mahalanobis')
        elif state.op == 'Weight':
            topLayer = LayerWeight(rng, n_inp=state.n_features, tag=' Weight')
        else:
            topLayer = Layer()

    apply_fn = eval(state.apply_fn)
    trainfunc = trainFnMember(apply_fn, topLayer, botLayer=botLayer, marge=state.marge)


    out = []
    outb = []
    outc = []

    state.bestout = np.inf

    _log.info('BEGIN TRAINING')
    timeref = time.time()
    for epoch_count in range(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(state.n_links)
        trainIdxl = state.trIdxl[order]
        trainIdxr = state.trIdxr[order]

        listidx = np.arange(state.n_samples, dtype='int32')
        listidx = listidx[np.random.permutation(len(listidx))]
        trainIdxrn = listidx[np.arange(state.n_links) % len(listidx)]
        trainIdxln = listidx[np.arange(state.n_links) % len(listidx)]


        for _ in range(20):
            outtmp = trainfunc(trainIdxl, trainIdxr, trainIdxln, trainIdxrn, state.lr_params)
            out += [outtmp[0]]
            outb += [outtmp[1]]
            outc += [outtmp[2]]
                # mapping.normalize()

            if np.mean(out) <= state.bestout:
                state.bestout = np.mean(out)
                state.lr_params *= 1.01
            else:
                state.lr_params *= .1

        if (epoch_count % state.neval) == 0:
            _log.info('-- EPOCH %s (%s seconds per epoch):' % (epoch_count, (time.time() - timeref) / state.neval))
            _log.info('Cost mean: %s +/- %s      updates: %s%% ' % (np.mean(out), np.std(out), np.mean(outb) * 100))
            _log.debug('Learning rate: %s LeaveOneOut: %s' % (state.lr_params, np.mean(outc)))

            timeref = time.time()

            ph_activate = botLayer()
            Pr = apply_fn(topLayer(ph_activate)).eval()
            state.train = np.mean(RankScoreIdx(Pr, state.trIdxl, state.trIdxr))
            _log.debug('Training set Mean Rank: %s  Score: %s' % (state.train, np.mean(Pr[state.trIdxr, state.trIdxl])))

            ph_activate = botLayer.layerout(state.X)
            Pr = apply_fn(topLayer(ph_activate)).eval()
            state.test = np.mean(RankScoreIdx(Pr, state.teIdxl, state.teIdxr))
            _log.debug('Testing set Mean Rank: %s  Score: %s' % (state.test, np.mean(Pr[state.teIdxr, state.teIdxl])))
            state.cepoch = epoch_count
            f = open(state.savepath + '/' + 'model' + '.pkl', 'wb')  # + str(state.cepoch)
            pickle.dump(state, f, -1)
            f.close()
            _log.debug('The saving took %s seconds' % (time.time() - timeref))
            timeref = time.time()

        outb = []
        outc = []
        out = []
        state.bestout = np.inf
        if state.lr_params < state.base_lr:      # if the learning rate is not growing
            state.base_lr *= 0.1

        state.lr_params = state.base_lr
        f = open(state.savepath + '/' + 'state.pkl', 'wb')
        pickle.dump(state, f, -1)
        f.close()

def launch(lr_rbm=0.01, rbm_training_epochs=15, totepochs=200, n_out=30, neval=10,
           dataname='webkb', rbm_batch_size=20, savepath='Output', datapath='./',
           n_chains=20, apply_fn='softmax', marge_ratio=5., seed=213, pretrain=True,
           n_hidden=30, lr_params=1000., op='None'):
    # configure logging
    FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    _log = logging.getLogger(dataname + ' experiment')
    _log.setLevel(logging.DEBUG)
    ch_file = logging.FileHandler(filename='predict_' + 'marge' +
                                           str(marge_ratio) + '.log', mode='w')
    ch_file.setLevel(logging.DEBUG)
    ch_file.setFormatter(FORMAT)
    ch = logging.StreamHandler()
    ch.setFormatter(FORMAT)
    ch.setLevel(logging.DEBUG)
    _log.addHandler(ch)
    _log.addHandler(ch_file)
    # set input and state variable
    _log.info('Start')

    state = DD()
    state.lr_rbm = lr_rbm
    state.lr_params = lr_params
    state.base_lr = lr_params
    state.rbm_training_epochs = rbm_training_epochs
    state.dataname = dataname
    state.rbm_batch_size = rbm_batch_size
    state.savepath = savepath
    state.n_chains = n_chains
    state.n_hidden = n_hidden
    state.marge_raio = marge_ratio
    state.seed = seed
    state.totepochs = totepochs
    state.neval = neval
    state.n_out = n_out
    state.pretrain = pretrain
    state.datapath = datapath


    if state.savepath not in os.listdir('./'):
        os.mkdir(state.savepath)

    # load the parsed data file
    f = open(datapath + dataname + '/' + dataname + '.pkl', 'rb')
    state.X = theano.shared(value=np.array(pickle.load(f), dtype=theano.config.floatX), borrow=True)
    state.train_set = theano.shared(value=np.array(pickle.load(f), dtype=theano.config.floatX), borrow=True)
    state.test_set = theano.shared(value=np.array(pickle.load(f), dtype=theano.config.floatX), borrow=True)
    state.trIdxl = np.asarray(pickle.load(f), dtype='int32')  # numpy indexes start from 0
    state.trIdxr = np.asarray(pickle.load(f), dtype='int32')
    state.teIdxl = np.asarray(pickle.load(f), dtype='int32')  # numpy indexes start from 0
    state.teIdxr = np.asarray(pickle.load(f), dtype='int32')
    f.close()

    state.n_samples, state.n_features = np.shape(state.train_set.get_value(borrow=True))
    state.n_links = np.shape(state.trIdxl)[0]
    state.marge = state.marge_raio / state.n_samples
    state.apply_fn = apply_fn
    state.op = op

    SGDexp(state, _log)

if __name__ == '__main__':
    launch()