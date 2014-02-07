#--------------------------------------------------------------#
# Data processing											   #
# Machine learning and validation and testing		 		   #
# Author : Ardo												   #
#--------------------------------------------------------------#

# import os.path
import json
import Stats
import ML
from numpy import *
import random

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

def loadFromFile(i,j,index,il,inMat,sumOfTestData,cls,currClass,prefix='fn'):
	fName = prefix+str(j)+'_'+str(i)+'.txt'
	fn = open(fName)
	lineNum=len(fn.readlines())
	sumOfTestData+=lineNum
	fn = open(fName)
	for line in range(0,lineNum):
		line=fn.readline()
		line=json.loads(line)
		inMat[index,:]=line[0:-1]
		if cls==1:
			il.append(line[-1])
		else:
			if line[-1]==currClass:
				il.append(1)
			else:
				il.append(cls)
		index+=1
	fn.close()
	return sumOfTestData,index

def loadKFCV(K,mlMethod,cls=1,currClass=''):
	testDataSize=stats.totalData/K+K
	for i in range(K):
		numClass=len(stats.classLabel)
		#load test data
		testMat=zeros((testDataSize,stats.numOfFeat))
		tl=[]
		index=0
		sumOfTestData=0
		for j in range(1,numClass+1):
			sumOfTestData,index=loadFromFile(i,j,index,tl,testMat,\
				sumOfTestData,cls,currClass)
		if sumOfTestData!=testDataSize:
			testMat=testMat[0:sumOfTestData,:]

		#load training data
		learnMat=zeros((testDataSize*(K-1),stats.numOfFeat))
		ll=[]
		index=0
		sumOfTestData=0
		for l in range(K):
			if l==i:
				continue
			for j in range(1,numClass+1):
				sumOfTestData,index=loadFromFile(l,j,index,ll,learnMat,\
					sumOfTestData,cls,currClass)
		if sumOfTestData!=testDataSize:
			learnMat=learnMat[0:sumOfTestData,:]

		print testMat.shape
		print learnMat.shape
		ml=ML.ML(learnMat,testMat,ll,tl)
		ml.valMethod='K-Fold CV'
		ml.iterations=i
		if mlMethod=='kNN':
			ml.kNN(20)
		elif mlMethod=='adaBoost':
			ml.valMethod='KFoldCV-1VSall'
			ml.currClass=currClass
			ml.classLabel=[klas[0] for klas in stats.classLabel]
			ml.adaBoost(70)
		else:
			print 'Default method: 3NN'
			ml.kNN(3)

def load1vsAll(mlMethod,K=10):
	numClass=len(stats.classLabel)
	for klas in stats.classLabel:
		print klas
		loadKFCV(K,mlMethod,-1,klas[0])
	
def load1FoldDataForTest():
	testDataSize=stats.totalData/10+10
	numClass=len(stats.classLabel)
	#load test data
	testMat=zeros((testDataSize,stats.numOfFeat))
	tl=[]
	index=0
	sumOfTestData=0
	for j in range(1,numClass+1):
		i=random.randrange(10)
		sumOfTestData,index=loadFromFile(i,j,index,tl,testMat,\
			sumOfTestData,1,'') #,'tf')
	if sumOfTestData!=testDataSize:
		testMat=testMat[0:sumOfTestData,:]
	print testMat.shape
	ml=ML.ML(testMat,testMat,tl,tl)
	ml.classLabel=[klas[0] for klas in stats.classLabel]
	ml.loadResult()
	fin=ml.adaBoostMultiClassClassify()



stats=loadStats()
# loadKFCV(10,'kNN')

load1vsAll('adaBoost')

# load1FoldDataForTest()


