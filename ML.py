#--------------------------------------------------------------#
# Machine Learning class									   #
# Various machine learning implementation 					   #
# Author : Ardo												   #
#--------------------------------------------------------------#
from numpy import *
import operator

class ML:

	def __init__(self,im,tm,il,tl):
		self.inputMatrix=im
		self.testMatrix=tm
		self.inputLabel=il
		self.testLabel=tl

	def kNNClassify(self,inM,K):
		dataSetSize = self.inputMatrix.shape[0]
		diffMat = tile(inM, (dataSetSize,1)) - self.inputMatrix
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		distances = sqDistances**0.5
		sortedDistIndicies = distances.argsort()     
		classCount={}          
		for i in range(K):
		    voteIlabel = self.inputLabel[sortedDistIndicies[i]]
		    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedClassCount[0][0]

	def kNN(self,K):
		retList=[]
		i=0; error=0
		for row in self.testMatrix:
			res=self.kNNClassify(row,K)
			retList.append(res)
			if res!=self.testLabel[i]:
				error+=1
			i=i+1
		# 	if i>50:
		# 		break
		# print retList
		print "Test Data: "
		print self.testMatrix.shape
		print "Error: "+str(error)+" "+str(error/float(self.testMatrix.shape[0]))+"  K="+str(K)


