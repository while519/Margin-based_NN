from scipy.io.matlab import loadmat
import numpy as np
import pickle

np.random.seed(123)
dataname = 'webkb'
datapath = '../Data/'

assert datapath is not None
mat = loadmat(datapath + dataname)

X = mat['X']
I = mat['I'] -1
X_samples, X_dims = X.shape
order = np.random.permutation(X_samples)
mapping_dic = dict(zip(order, np.arange(X_samples)))
n_train = round(X_samples * 0.9)
n_test = X_samples - n_train

train_x = X[order[:n_train], :]
test_x = X[order[n_train :], :]

trIdxl = []
trIdxr = []
teIdxl = []
teIdxr = []
for idxl, idxr in zip(I[:,0], I[:,1]):
    if (mapping_dic[idxl] < n_train and mapping_dic[idxr] < n_train):
        trIdxl += [mapping_dic[idxl]]
        trIdxr += [mapping_dic[idxr]]
    else:
        teIdxl += [idxl]
        teIdxr += [idxr]

f = open('./' + dataname + '.pkl', 'wb')
pickle.dump(X, f, -1)
pickle.dump(train_x, f, -1)
pickle.dump(test_x, f, -1)
pickle.dump(trIdxl, f, -1)
pickle.dump(trIdxr, f, -1)
pickle.dump(teIdxl, f, -1)
pickle.dump(teIdxr, f, -1)
f.close()
