import gzip, cPickle
import theano
import theano.tensor as T
import numpy as np

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def Load_data(dataset):
    print '... loading data'
    train_set, valid_set, test_set = cPickle.load( gzip.open(dataset, 'rb') )
    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y   = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def Load_data_ind(dataset):
    print '... loading data'
    test_set = cPickle.load( gzip.open(dataset, 'rb') )
    test_set_x, test_set_y   = shared_dataset(test_set)
    return [(test_set_x, test_set_y)]

def Load_npdata(dataset):
    print '... loading data'
    datasets = np.load(dataset)
    test_setx = datasets['test_seq']
    test_sety = datasets['test_lab']
    test_set = (test_setx, test_sety)
    test_set_x, test_set_y   = shared_dataset(test_set)
    return [(test_set_x, test_set_y)]

