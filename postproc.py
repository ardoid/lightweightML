import json

def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m

def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t

# [1 if p < .5 else 2 for p in classifications]
f=open('output A.txt')
a=f.readline()
b=f.readline()
a=json.loads(a)
b=json.loads(b)
c=[]
d=[]
idx=0
for e in a:
	if e=='sitting':
		c.append(1)
	elif e=='sittingdown':
		c.append(2)
	elif e=='standing':
		c.append(3)
	elif e=='standingup':
		c.append(4)
	elif e=='walking':
		c.append(5)
	else:
		c.append(0)

for e in b:
	if e=='sitting':
		d.append(1)
	elif e=='sittingdown':
		d.append(2)
	elif e=='standing':
		d.append(3)
	elif e=='standingup':
		d.append(4)
	elif e=='walking':
		d.append(5)
	else:
		d.append(0)
print len(d)
print len(c)
print create_conf_matrix(d,c,6)


