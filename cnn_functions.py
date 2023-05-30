import theano
import theano.tensor as T
import os, sys, timeit, gzip, glob, numpy, math, cPickle

from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from collections import OrderedDict
from load_data import Load_data
from mlp import relu, HiddenLayer
from pyroc import *

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in  = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        W_bound = numpy.sqrt(10. / (fan_in + fan_out))  
        self.W  = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),  borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b   = theano.shared(value=b_values, borrow=True)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),name='W',borrow=True
            )
        if b is None:
            b = theano.shared(
                value=numpy.zeros((n_out,),
                    dtype=theano.config.floatX
                ),name='b',borrow=True
            )
        self.W = W
        self.b = b
        self.output = T.nnet.sigmoid(T.dot(input, self.W)+ self.b).flatten()
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W)+ self.b).flatten()
        self.y_pred = self.output > .5
        self.params = [self.W, self.b]
        self.x = input
    def negative_log_likelihood(self, y):
        return - T.mean(y * T.log(self.output) + (1 - y) * T.log(1. - self.output))

class CNN(object):
    def __init__(self, rng=None, nkerns=None, batch_size=None, in_dim=None, filtsize=None, poolsize=None, hidden=None):
        self.layers = []
        self.params = []
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        layer0_input = self.x.reshape((batch_size, 1, in_dim[0], in_dim[1]))
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, in_dim[0], in_dim[1]),
            filter_shape=(nkerns[0], 1, filtsize[0][0], filtsize[0][1]),
            poolsize=poolsize[0]
        )
        self.layers.append(layer0)
        dim11 = (in_dim[0]-filtsize[0][0] +1)
        dim12 = (in_dim[1]-filtsize[0][1] +1)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], dim11, dim12),
            filter_shape=(nkerns[1], nkerns[0], filtsize[1][0], filtsize[1][1]),
            poolsize=poolsize[1]
        )
        self.layers.append(layer1)
        dim21 = (dim11 - filtsize[1][0] +1)
        dim22 = (dim12 - filtsize[1][1] +1)
        layer2_input = layer1.output.flatten(2)
        layer2 = LogisticRegression(input=layer2_input, n_in=hidden, n_out=1)
        self.layers.append(layer2)
        self.params = [ param for layer in self.layers for param in layer.params ]
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        self.finetune_cost = layer2.negative_log_likelihood(self.y)
    def build_finetune_functions(self, datasets, batch_size, learning_rate, L1_param, L2_param, mom):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]
        index = T.lvector('index')
        gparams = T.grad( self.finetune_cost  + L1_param * abs(self.layers[-1].W).sum() + L2_param * (self.layers[-1].W **2).sum(), self.params)
        updates1 = OrderedDict()
        for param, gparam, gparam_mom in zip(self.params, gparams, self.gparams_mom):
            updates1[gparam_mom] = mom * gparam_mom - learning_rate * gparam
            updates1[param] = param + updates1[gparam_mom]
        train_fn = theano.function(
            inputs =[index],
            outputs=self.finetune_cost,
            updates=updates1,
            givens={ self.x: train_set_x[index],
                     self.y: train_set_y[index]}  )
        valid_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: valid_set_x[index]} )
        valid_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: valid_set_y[index] } )
        test_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: test_set_x[index]} )
        test_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: test_set_y[index] } )
        def getVals( fn, IDX, n_exp, batch_size ):
            vals = list()
            n_batches = n_exp/ batch_size
            resid     = n_exp  - (n_batches * batch_size)
            cnt = int(math.ceil(batch_size / n_exp))
            for i in range(n_batches):
                vals+= fn(IDX[i*batch_size:(i+1)*batch_size]).tolist()
            if cnt <= 1 and resid !=0:
                val = fn(IDX[(n_batches-1)*batch_size+resid:(n_batches*batch_size)+resid])
                vals+= val[(batch_size-resid):batch_size].tolist()
            if cnt > 1:
                IDX_ = IDX
                for i in range(cnt-1):
                    IDX_ = numpy.concatenate((IDX_, IDX))
                val = fn(IDX_[0: batch_size])
                vals += val[range(n_exp)].tolist()
            return vals
        n_valid_exp = valid_set_x.get_value(borrow=True).shape[0]
        n_test_exp  =  test_set_x.get_value(borrow=True).shape[0]
        def valid_check():
            idx = numpy.random.permutation(range(n_valid_exp))
            valid_y    = getVals( valid_y_fn,   idx, n_valid_exp, batch_size )
            valid_pred = getVals( valid_pred_fn,idx, n_valid_exp, batch_size )
            return valid_y, valid_pred
        def test_check():
            idx = numpy.random.permutation(range(n_test_exp))
            test_y     = getVals( test_y_fn,    idx, n_test_exp, batch_size )
            test_pred  = getVals( test_pred_fn, idx, n_test_exp, batch_size )
            return test_y, test_pred
        return train_fn, valid_check, test_check

