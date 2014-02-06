#--------------------------------------------------------------#
# Data preprocessing										   #
# Do all kind of filtering and normalization 				   #
# Author : Ardo												   #
#--------------------------------------------------------------#

import csv
import json
import os.path
import sys
import locale
import random

locale.setlocale(locale.LC_ALL,'de_DE') # 'en_US.UTF-8')
statsFile    = 'stats.txt'
inputFile    = 'dataset.csv'
datatypeFile = 'datatype.csv'

def splitDataByClass(inputFile):
	listOfClass=[]; numOfClass=[]; typeOfClass=[]
	maxVal=[]; minVal=[]
	outputFile  = []
	with open(datatypeFile,'rb') as dt:
		reader=csv.reader(dt,delimiter=';')
		typeOfClass=reader.next()
		print typeOfClass
	dt.close()
	featNum=len(typeOfClass)-1
	#init min and max values for the features
	for typ in typeOfClass:
		if typ=='string':
			minVal.append('a')
			maxVal.append('z')
		elif typ=='int':
			maxVal.append(-sys.maxint-1)
			minVal.append(sys.maxint)
		elif typ=='float':
			maxVal.append(-float('inf'))
			minVal.append(float('inf'))
		else:
			minVal.append('x')
			maxVal.append('x')			

	with open(inputFile, 'rb') as f:
		reader = csv.reader(f,delimiter=';')
		heading = reader.next()
		# count = 0
		print heading
		for row in reader:
			for i in range(0,featNum):
				if typeOfClass[i]=='int':
					row[i]=int(row[i])					
				elif typeOfClass[i]=='float':
					row[i]=locale.atof(row[i])
				if typeOfClass[i]=='int' or typeOfClass[i]=='float':
					if row[i]<minVal[i]:
						minVal[i]=row[i]
					elif row[i]>maxVal[i]:
						maxVal[i]=row[i]	
			if row[-1] not in listOfClass:
				listOfClass.append(row[-1])
				numOfClass.append(1)
				fName = 'o'+str(len(numOfClass))+'.txt'
				output = open(fName,'w')
				outputFile.append(output)
			else:
				indexOfClass = listOfClass.index(row[-1])
				outputFile[indexOfClass].write(json.dumps(row))
				outputFile[indexOfClass].write('\n')
				numOfClass[indexOfClass]=numOfClass[indexOfClass]+1
			# count = count + 1
			# if count==1000:
			# 	break
		output = open(statsFile,'w')
		output.write('Class\t\t\t Size\n')
		output.write('------\t\t\t-----\n')
		idx = 0
		sums = 0
		for cl in listOfClass:
			output.write(cl+'\t\t\t'+str(numOfClass[idx])+'\n')
			sums = sums + numOfClass[idx]
			idx = idx+1
		output.write('Total: '+str(sums)+'\n')
		output.write('------------------------------------\n')
		output.write('Min values:\n')
		output.write(json.dumps(minVal))
		output.write('\nMax values:\n')
		output.write(json.dumps(maxVal))
		for x in typeOfClass:
			if x=='string' or x=='class':
				featNum=featNum-1
		output.write('\nFeatures: '+str(featNum))

def splitDataKfoldCV(K):
	#read datatype file to know the feature datatype
	dt = open(datatypeFile,'rb')
	typeOfClass=dt.readline().split(';')
	dt.close()

	#read the stats file for number of test data, min max value
	stats = open(statsFile,'rb')
	fileLen=len(stats.readlines())
	stats = open(statsFile,'rb')
	stats.readline()
	stats.readline()
	data = []
	for i in range(0,fileLen):
		line=stats.readline()
		data.append(line.strip().split('\t\t\t'))
		if line.split(' ')[0]=='Total:':
			break
	stats.readline()
	stats.readline()
	minVal=json.loads(stats.readline())
	stats.readline()
	maxVal=json.loads(stats.readline())
	stats.close()
	data.pop() #remove the Total

	classIdx = 1
	for d in data:
		d[1]=int(d[1])
		print d
		foldSizeRem = d[1]%K
		foldSize = d[1]/K + 1
		print str(foldSize)
		f = open('o'+str(classIdx)+'.txt')
		outputFile = []; outputNormFile = []
		numOfClass = [0 for x in range(K)]
		for k in range(0,K):
			fName = 'fn'+str(classIdx)+'_'+str(k)+'.txt'
			output = open(fName,'w')
			outputFile.append(output)
			fName = 'tf'+str(classIdx)+'_'+str(k)+'.txt'
			output = open(fName,'w')
			outputNormFile.append(output)
		for line in f:
			nextIdx = random.randrange(K)
			count = 0
			while numOfClass[nextIdx] > foldSize and count < K:
				nextIdx = nextIdx + 1
				if nextIdx >= K:
					nextIdx = 0
				count = count + 1
			if count > K:
				#create error log file here
				print 'Warning: There is feature data not accounted for\n'
			else:
				#normalize the feature values
				unNormData=json.loads(line)
				normData=[]
				for x,y,mn,mx in zip(unNormData,typeOfClass,minVal,maxVal):
					if y=='int':
						normData.append((x-mn)/float(mx-mn))
					elif y=='float':
						normData.append((x-mn)/float(mx-mn))
					elif y=='sex':						
						if x=='Woman' or x=='Female':
							normData.append(0)
						elif x=='Man' or x=='Male':
							normData.append(1)
					elif y=='string':
						pass
					# 	normData.append(x)
					else:
						normData.append(x)
				outputNormFile[nextIdx].write(line)
				outputFile[nextIdx].write(json.dumps(normData)+'\n')
				numOfClass[nextIdx]=numOfClass[nextIdx]+1
		classIdx=classIdx+1

def normalizeInput():
	pass


#start of the preprocessing program
if not (os.path.isfile(statsFile)):
	splitDataByClass(inputFile)
else:
	splitDataKfoldCV(10)









