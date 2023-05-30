import theano
import theano.tensor as T
import numpy as np
import sys, gzip, cPickle, subprocess, math

from cnn_functions import LeNetConvPoolLayer, CNN, LogisticRegression
from load_data import Load_data
from load_data import Load_data_ind

modelFile  = sys.argv[1]
testdata = sys.argv[2]
predFile  = sys.argv[3]

#print '\n', 'Model file: ', modelFile, '\n'
#print 'Test data: ', testdata, '\n'
#print 'Prediction result: ', predFile, '\n'

classifier	= cPickle.load(gzip.open(modelFile))
datasets = Load_data_ind(testdata)
test_set_x, test_set_y = datasets[0]

get_y	= theano.function([], test_set_y)
y_      = get_y()
x_      = np.asarray(test_set_x.get_value(borrow=True) , dtype='float32')

batch_size=int(10)
predict_model = theano.function( inputs = [classifier.x], outputs= classifier.layers[-1].output )
n_exp = ( x_.shape[0] )
cnt = int(math.ceil(batch_size / n_exp))
n_batches = n_exp / batch_size
resid = n_exp - (n_batches * batch_size)
y_answer = list()
y_pred = list()
for index in range(n_batches):
	xx = x_[index * batch_size: (index + 1) * batch_size]
	res = predict_model(xx)
	y_pred += res[range(batch_size)].tolist()

if cnt <= 1 and resid != 0:
	xx = x_[(n_batches-1) * batch_size + resid: (n_batches*batch_size)+resid]
	res = predict_model(xx)
	y_pred += res[(batch_size-resid):batch_size].tolist()

if cnt > 1:
	xx = x_
	for i in range(cnt-1):
		xx = np.concatenate((xx, x_))
	res = predict_model(xx[0: batch_size])
	y_pred += res[range(n_exp)].tolist()

fout = open(predFile,'w')
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('/')[-1].split('.')[0]+'.'+testdata.split('/')[-1].split('.')[1]).readlines()]
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('.')[0]+'.'+testdata.split('.')[1]).readlines()]
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata).readlines()]
tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('/')[-1].split('.')[0]+'.'+testdata.split('/')[-1].split('.')[1]).readlines()]
#print(tids)
for i in range(len(y_)):
#        print(tids[i])
	fout.write(tids[i]+'\t'+str(y_pred[i])+'\n')
fout.close()


