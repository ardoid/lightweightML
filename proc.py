#--------------------------------------------------------------#
# Data processing											   #
# Machine learning and validation and testing		 		   #
# Author : Ardo												   #
#--------------------------------------------------------------#

# import json
import os.path
import sys
import json
import random
import Stats
import ML
from numpy import *

resultFile    = 'result.csv'

def loadStats():
	stats = Stats.Stats()
	return stats

def writeData():
	pass

def testData(method,param,mlMethod):	
	if method=='KFCV':
		loadKFCV(param,mlMethod)
	elif method=='1VSALL':
		load1vsAll()
	else:
		print 'Error: Unknown method'

def loadFromFile(i,j):
	fName = 'fn'+str(j)+'_'+str(i)+'.txt'
	fn = open(fName)
	lineNum=len(fn.readlines())
	sumOfTestData+=lineNum
	fn = open(fName)
	for line in range(0,lineNum):
		line=fn.readline()
		line=json.loads(line)
		learnMat[index,:]=line[0:-1]
		tl.append(line[-1])
		index+=1
	fn.close()
	if sumOfTestData!=testDataSize:
		learnMat=learnMat[0:sumOfTestData,:]


def loadKFCV(K,mlMethod):
	testDataSize=stats.totalData/K+K
	for i in range(3,4):
		numClass=len(stats.classLabel)
		#load test data
		testMat=zeros((testDataSize,stats.numOfFeat))
		tl=[]
		index=0
		sumOfTestData=0
		for j in range(1,numClass+1):
			fName = 'fn'+str(j)+'_'+str(i)+'.txt'
			fn = open(fName)
			lineNum=len(fn.readlines())
			sumOfTestData+=lineNum
			fn = open(fName)
			for line in range(0,lineNum):
				line=fn.readline()
				line=json.loads(line)
				testMat[index,:]=line[0:-1]
				tl.append(line[-1])
				index+=1
			fn.close()
		if sumOfTestData!=testDataSize:
			testMat=testMat[0:sumOfTestData,:]

		#load training data
		learnMat=zeros((testDataSize*(K-1),stats.numOfFeat))
		# print learnMat.shape
		ll=[]
		index=0
		sumOfTestData=0
		for l in range(i+1,K):
			for j in range(1,numClass+1):
				fName = 'fn'+str(j)+'_'+str(l)+'.txt'
				fn = open(fName)
				lineNum=len(fn.readlines())
				sumOfTestData+=lineNum
				fn = open(fName)
				for m in range(0,lineNum):
					line=fn.readline()
					line=json.loads(line)
					learnMat[index,:]=line[0:-1]
					ll.append(line[-1])
					index+=1
				fn.close()
		if sumOfTestData!=testDataSize:
			learnMat=learnMat[0:sumOfTestData,:]

		print testMat.shape
		print learnMat.shape
		ml=ML.ML(learnMat,testMat,ll,tl)
		if mlMethod=='kNN':
			ml.kNN(7)
		else:
			print 'Default method: 3NN'
			ml.kNN(3)

def load1vsAll():
	numClass=len(stats.classLabel)

	pass

stats=loadStats()
loadKFCV(10,'kNN')
