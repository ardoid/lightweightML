#--------------------------------------------------------------#
# Machine Learning class									   #
# Various machine learning implementation 					   #
# Author : Ardo												   #
#--------------------------------------------------------------#
from numpy import *
import operator
import json

class ML:

	valMethod=' '
	iterations=0
	currClass=' '
	classLabel=[]
	finalClassifier=[]
	resultFile='result.csv'

	def __init__(self,im,tm,il,tl):
		self.inputMatrix=im
		self.testMatrix=tm
		self.inputLabel=il
		self.testLabel=tl

	def logResult(self,toWrite,classArr=[]):
		print toWrite
		isEmpty=False
		with open(self.resultFile) as csvfile:
			if len(csvfile.readlines())==0:
				isEmpty=True
		with open(self.resultFile,'a') as csvfile:
			if isEmpty:
				header=['Method','Class','K-Fold','Input Data','Test Data','Features',\
				'Error','Error %','ML method','HyperParameter','Remark','HP details']
				s=';'.join(header)
				csvfile.write(s+'\n')
			s=';'.join(toWrite)
			print s
			csvfile.write(s)
			csvfile.write('\n')

	def loadResult(self):
		print self.classLabel
		for klas in self.classLabel:
			# with open(self.resultFile) as csvfile:
			csvfile=open(self.resultFile)
			filelen=len(csvfile.readlines())-1
			csvfile=open(self.resultFile)
			csvfile.readline()
			errMargin=100.0
			bestClassif=0
			for i in range(filelen):
				line=csvfile.readline()
				line=line.split(';')
				if line[1]==klas and float(line[7])<errMargin:
					errMargin=float(line[7])
					bestClassif=json.loads(line[11])
			self.finalClassifier.append(bestClassif)

	def kNNClassify(self,inM,K):
		dataSetSize=self.inputMatrix.shape[0]
		diffM=tile(inM, (dataSetSize,1)) - self.inputMatrix
		sqDiffM=diffM**2
		sqDist=sqDiffM.sum(axis=1)
		distances=sqDist**0.5
		sortedDistIndicies=distances.argsort()     
		classCnt={}          
		for i in range(K):
		    voteLabel=self.inputLabel[sortedDistIndicies[i]]
		    classCnt[voteLabel]=classCnt.get(voteLabel,0) + 1
		sClassCount=sorted(classCnt.iteritems(),key=operator.itemgetter(1),reverse=True)
		return sClassCount[0][0]

	def kNN(self,K=3):
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
		toFile=[self.valMethod,' ',str(self.iterations),str(self.inputMatrix.shape[0]),\
				str(self.testMatrix.shape[0]),str(self.inputMatrix.shape[1]),\
				str(error),str(errPct),'KNN',str(K)]
		self.logResult(toFile)

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
	        for j in range(-1,int(numSteps)+1):
	            for inequal in ['lt', 'gt']:
	                threshVal=(rangeMin + float(j) * stepSize)
	                predictedVals=self.decisionStump(self.inputMatrix,i,threshVal,inequal)
	                errArr=mat(ones((m,1)))
	                errArr[predictedVals==labelMat] = 0
	                weightedError=D.T*errArr  #calc total error multiplied by D
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
	        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
	        bestStump['alpha']=alpha  
	        weakClassArr.append(bestStump) 
	        expon = multiply(-1*alpha*classLabels.T,classEst)
	        D=multiply(D,exp(expon)) 
	        D=D/D.sum()
	        aggClassEst+=alpha*classEst
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
	        aggClassEst+=classifierArr[i]['alpha']*classEst
	    return sign(aggClassEst)

	def adaClassifyTest(self,testMatrix,classifierArr,hP):
	    m=shape(testMatrix)[0]
	    aggClassEst=mat(zeros((m,1)))
	    for i in range(len(classifierArr)):
	        classEst=self.decisionStump(testMatrix,\
	        						 classifierArr[i]['dim'],\
	                                 classifierArr[i]['thresh'],\
	                                 classifierArr[i]['ineq'])
	        aggClassEst+=classifierArr[i]['alpha']*classEst

		testLabelM=mat(self.testLabel)
	    aggErrors=multiply(sign(aggClassEst)!=testLabelM.T,ones((m,1)))
	    errorRate=aggErrors.sum()/m * 100
	    print m
	    print "Error Rate: "+str(errorRate)
	    toFile=[self.valMethod,self.currClass,str(self.iterations),str(self.inputMatrix.shape[0]),\
	    		str(self.testMatrix.shape[0]),str(self.inputMatrix.shape[1]),\
	    		str(aggErrors.sum()),str(errorRate),'AdaBoost',str(hP),'Validation stage',json.dumps(classifierArr)]
	    self.logResult(toFile)

	def adaBoost(self,hP=40):
		weakClassArr,aggClassEst=self.adaBoostTrain(hP)
		self.adaClassifyTest(self.testMatrix,weakClassArr,hP)

	def adaBoostMultiClassClassify(self):
		classIndex=0
		finalResult=[None]*self.testMatrix.shape[0]
		for klas in self.classLabel:
			result=self.adaClassify(self.testMatrix,self.finalClassifier[classIndex])
			index=0
			for res in result:
				if res==1:
					finalResult[index]=klas
				index+=1
			classIndex+=1
# [i for i, j in zip(a, b) if i == j]
		print self.testMatrix.shape
		m=shape(self.testMatrix)[0]
		testLabelM=mat(self.testLabel)
		finalResultM=mat(finalResult)
		aggErrors=multiply(finalResultM!=testLabelM,ones((1,m)))
		errorRate=aggErrors.sum()/m * 100
		with open('finres.txt','w') as fr:
			fr.write(json.dumps(finalResult))
			fr.write('\n')
			fr.write(json.dumps(self.testLabel))
		print str(errorRate)		
		return finalResult
		




