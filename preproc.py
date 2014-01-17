#--------------------------------------------------------------
# Data preprocessing
# Author : Ardo
#--------------------------------------------------------------

import csv
import json
import os.path
import random

statsFile = 'stats.txt'
inputFile = 'dataset.csv'

def splitDataByClass(inputFile):
	listOfClass = []
	numOfClass  = []
	outputFile  = []
	with open(inputFile, 'rb') as f:
		reader = csv.reader(f,delimiter=';')
		heading = reader.next()
		# count = 0
		print heading
		for row in reader:
			# print row
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
		output.write('Total: '+str(sums))

def splitDataKfoldCV(K):
	stats = open(statsFile,'rb')
	stats.readline()
	stats.readline()
	data = []
	for line in stats:
		data.append(line.strip().split('\t\t\t'))
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
		outputFile = []
		numOfClass = [0 for x in range(K)]
		for k in range(0,K):
			fName = 'f'+str(classIdx)+'_'+str(k)+'.txt'
			output = open(fName,'w')
			outputFile.append(output)
		for line in f:
			nextIdx = random.randrange(K)
			count = 0
			while numOfClass[nextIdx] > foldSize and count < 10:
				nextIdx = nextIdx + 1
				if nextIdx >= K:
					nextIdx = 0
				count = count + 1
			if count > 9:
				print 'Oh No'
			else:
				outputFile[nextIdx].write(line)
				# outputFile[indexOfClass].write('\n')
				numOfClass[nextIdx]=numOfClass[nextIdx]+1
		classIdx=classIdx+1




#start of the preprocessing program
if not (os.path.isfile(statsFile)):
	splitDataByClass(inputFile)
else:
	splitDataKfoldCV(10)









