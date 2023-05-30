#!/usr/bin/python
import theano
import os, sys, gzip, cPickle
import numpy as np

def matchDat(afflst, hladic, aadic):
	seqlst = []
	tablst = []
	header = []
	for affin in afflst:
		affstr = affin.strip().split('\t')
		if affstr[0] in hladic:
			hlaseq = hladic[affstr[0]]
			aaseq = affstr[1]
			tmp = []
			tmp0 = []
			for hlain in hlaseq:
				for aain in aaseq:
					if hlain == 'X' or aain=='X':
						tmp0.append([float(0)])
					elif hlain == '*':
						tmp0.append([float(0)])
					elif hlain == '.':
						tmp0.append([float(0)])
					elif aain == 'X':
						tmp0.append([float(0)])
					elif aain == 'U':
						tmp0.append([aadic[hlain, 'C']])
					elif aain == 'J':
						aa1 = aadic[hlain, 'L']
						aa2 = aadic[hlain, 'I']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					elif aain == 'Z':
						aa1 = aadic[hlain, 'Q']
						aa2 = aadic[hlain, 'E']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					elif aain  == 'B':
						aa1 = aadic[hlain, 'D']
						aa2 = aadic[hlain, 'N']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					else:
						tmp0.append([aadic[hlain, aain]])
				tmp.append(tmp0)
				tmp0 = []
			seqlst.append(zip(*tmp))
			tablst.append(int(affstr[2]))
			header.append((affstr[0], affstr[1]))
	seqarray0 = np.array(seqlst, dtype = theano.config.floatX)
	del seqlst
	a_seq2 = seqarray0.reshape(seqarray0.shape[0], seqarray0.shape[1] * seqarray0.shape[2])
	a_lab2 = np.array(tablst, dtype = theano.config.floatX)
	del tablst
	return ((a_seq2, a_lab2)), header
	del a_seq2, a_lab2, header

def HeaderOutput(lstin, outname):
	outw = open(outname, 'w')
	for lin in lstin:
		outw.write('\t'.join(lin)+'\n')
	outw.close()

def modifyMatrix(affydatin_test, seqdatin,outfile):
	hladicin = {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in open(seqdatin).readlines()}
	aalst = open('data/Calpha.txt').readlines()
	aadicin = {}
	aaseq0 = aalst[0].strip().split('\t')
	for aain in aalst[1:]:
		aastr = aain.strip().split('\t' )
		for i in range(1, len(aastr)):
			aadicin[aaseq0[i-1], aastr[0]] = float(aastr[i])
	afflst = open(affydatin_test).readlines()
	d, test_header = matchDat(afflst, hladicin, aadicin)
	outname0 = affydatin_test
	outname2 = affydatin_test+'.header'
	#np.savez_compressed(outname0, test_seq = test_seq, test_lab = test_lab)
        cPickle.dump(d, gzip.open(outfile, 'wb'), protocol = 2)
	HeaderOutput(test_header, outname2)

Datname = sys.argv[1]
mhcclass=sys.argv[2]
outputfile=sys.argv[3]
print '\nInput file: ', Datname, '\n'




if mhcclass=='class1' :
    modifyMatrix(Datname, 'data/All_prot_alignseq_C_369.dat',outputfile)
    print 'The running is completed!\n'


if mhcclass=='class2' :
    modifyMatrix(Datname, 'data/MHC2_prot_alignseq.dat',outputfile)
    print 'The running is completed!\n'

