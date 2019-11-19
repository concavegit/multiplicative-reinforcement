import numpy as np

def factorization(n):
	arr = []
	for i in range(n):
		onenumber = []
		onenumber.append(1 if i%2==0 else 0)
		onenumber.append(1 if i%3==0 else 0)
		onenumber.append(1 if i%5==0 else 0)
		onenumber.append(1 if i%7==0 else 0)
		onenumber.append(1 if i==1 else 0)
		onenumber.append(i)
		arr.append(onenumber)
	a = np.array(arr)
	return a
print(factorization(21))