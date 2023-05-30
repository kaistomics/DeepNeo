#!/usr/bin/python
import os, sys
mhc_class = sys.argv[1]
predtype= sys.argv[2]
Inputname = sys.argv[3]
Resultname = sys.argv[4]

if mhc_class == "class1"  and predtype== 'tcr'  :
    os.system('THEANO_FLAGS=mode=FAST_RUN,device=cuda3,floatX=float32 ' \
        +'python cnn.py ' \
        +'data/tcr1-pan.pkl.gz ' \
        + Inputname + ' ' \
        + Resultname)
    print ("\nThe running is completed!\n")
if mhc_class=="class1" and predtype=='mhc':
	os.system('THEANO_FLAGS=mode=FAST_RUN,device=cuda3,floatX=float32 python cnn.py data/mhc1-pan.pkl.gz '+Inputname+' '+Resultname)
	print("\nThe running is completed!\n")

if mhc_class=="class2" and predtype=='mhc':
	os.system('THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python cnn.py data/mhc2-pan.pkl.gz '+Inputname+' '+Resultname)
	print("\nThe running is completed!\n")
if mhc_class == "class2"  and predtype== 'tcr'  :
    os.system('THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 ' \
        +'python cnn.py ' \
        +'data/tcr2-pan.pkl.gz ' \
        + Inputname + ' ' \
        + Resultname)
    print ("\nThe running is completed!\n")

