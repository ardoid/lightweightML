#--------------------------------------------------------------#
# Machine Learning class									   #
# Various machine learning implementation 					   #
# Author : Ardo												   #
#--------------------------------------------------------------#
from numpy import *
import operator
import csv

class ML:

	valMethod=' '
	iterations=0
	resultFile='result.csv'

	def __init__(self,im,tm,il,tl):
		self.inputMatrix=im
		self.testMatrix=tm
		self.inputLabel=il
		self.testLabel=tl

	def kNNClassify(self,inM,K):
		dataSetSize = self.inputMatrix.shape[0]
		diffMat = tile(inM, (dataSetSize,1)) - self.inputMatrix
		sqDiffMat = diffMat**2
		sqDist = sqDiffMat.sum(axis=1)
		distances = sqDist**0.5
		sortedDistIndicies = distances.argsort()     
		classCnt={}          
		for i in range(K):
		    voteLabel = self.inputLabel[sortedDistIndicies[i]]
		    classCnt[voteLabel] = classCnt.get(voteLabel,0) + 1
		sClassCount = sorted(classCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sClassCount[0][0]

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
		errPct=error/float(self.testMatrix.shape[0])*100
		print "Test Data: "
		print self.testMatrix.shape
		print "Error: "+str(error)+" "+str(errPct)+"  K="+str(K)
		# toFile=[self.valMethod,str(self.iterations),str(self.inputMatrix.shape[0]),\
		# 		str(self.testMatrix.shape[0]),str(self.inputMatrix.shape[1]),\
		# 		str(error),str(errPct),str(K)]
		toFile=[self.valMethod,self.iterations,self.inputMatrix.shape[0],\
				self.testMatrix.shape[0],self.inputMatrix.shape[1],error,errPct,K]
		toFile=str(toFile)
		with open(self.resultFile,'a') as csvfile:
			logwriter=csv.writer(csvfile,delimiter=';')
			logwriter.writerow(toFile)

	def decisionStump(self,inputMatrix,dimen,threshVal,threshIneq):
	    retArray = ones((shape(inputMatrix)[0],1))
	    if threshIneq == 'lt':
	        retArray[inputMatrix[:,dimen] <= threshVal] = -1.0
	    else:
	        retArray[inputMatrix[:,dimen] > threshVal] = -1.0
	    return retArray
	    
	def buildStump(self,D):
	    labelMat=mat(self.inputLabel).T
	    m,n=shape(self.inputMatrix)
	    numSteps=10.0; bestStump={}; bestClasEst=mat(zeros((m,1)))
	    minError=inf
	    for i in range(n):
	        rangeMin=self.inputMatrix[:,i].min(); rangeMax = self.inputMatrix[:,i].max();
	        stepSize=(rangeMax-rangeMin)/numSteps
	        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
	            for inequal in ['lt', 'gt']:
	                threshVal=(rangeMin + float(j) * stepSize)
	                predictedVals=self.decisionStump(self.inputMatrix,i,threshVal,inequal)
	                errArr=mat(ones((m,1)))
	                errArr[predictedVals==labelMat] = 0
	                weightedError=D.T*errArr  #calc total error multiplied by D
#print "split: dim %d, thresh %.2f, thresh ineqal: %s, 
#the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
	                if weightedError < minError:
	                    minError=weightedError
	                    bestClasEst=predictedVals.copy()
	                    bestStump['dim']=i
	                    bestStump['thresh']=threshVal
	                    bestStump['ineq']=inequal
	    return bestStump,minError,bestClasEst

	def adaBoostTrain(self,numIt=40):
	    weakClassArr=[]
	    m=shape(self.inputMatrix)[0]
	    D=mat(ones((m,1))/m)
	    aggClassEst=mat(zeros((m,1)))
	    classLabels=mat(self.inputLabel)
	    for i in range(numIt):
	        bestStump,error,classEst=self.buildStump(D)
	        #print "D:",D.T
	        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
	        bestStump['alpha']=alpha  
	        weakClassArr.append(bestStump) 
	        #print "classEst: ",classEst.T
	        expon = multiply(-1*alpha*classLabels.T,classEst)
	        D=multiply(D,exp(expon)) 
	        D=D/D.sum()
	        aggClassEst+=alpha*classEst
	        #print "aggClassEst: ",aggClassEst.T
	        aggErrors=multiply(sign(aggClassEst)!=classLabels.T,ones((m,1)))
	        errorRate=aggErrors.sum()/m
	        if errorRate==0.0: break
	    print "total error: ",errorRate
	    return weakClassArr,aggClassEst

	def adaClassify(self,testMatrix,classifierArr):
	    m=shape(testMatrix)[0]
	    aggClassEst=mat(zeros((m,1)))
	    print classifierArr
	    for i in range(len(classifierArr)):
	        classEst=self.decisionStump(testMatrix,\
	        						 classifierArr[i]['dim'],\
	                                 classifierArr[i]['thresh'],\
	                                 classifierArr[i]['ineq'])
	        print classEst.shape
	        aggClassEst+=classifierArr[i]['alpha']*classEst
	    return sign(aggClassEst)

	def adaClassifyTest(self,testMatrix,classifierArr):
	    m=shape(testMatrix)[0]
	    aggClassEst=mat(zeros((m,1)))
	    print classifierArr
	    for i in range(len(classifierArr)):
	        classEst=self.decisionStump(testMatrix,\
	        						 classifierArr[i]['dim'],\
	                                 classifierArr[i]['thresh'],\
	                                 classifierArr[i]['ineq'])
	        print classEst.shape
	        aggClassEst+=classifierArr[i]['alpha']*classEst

		testLabelM=mat(self.testLabel)
		m=testLabelM.shape[0]
	    aggErrors=multiply(sign(aggClassEst)!=testLabelM.T,ones((m,1)))
	    errorRate=aggErrors.sum()/m
	    print str(errorRate)


	def adaBoost(self):
		weakClassArr,aggClassEst=self.adaBoostTrain()
		self.adaClassifyTest(self.testMatrix,weakClassArr)


