import numpy as np

def factorization(n):
	"""
	input a number to get a matrix describing factors of 2, 3, 5, 7, whether the number is 1, and the number itself.
	"""
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