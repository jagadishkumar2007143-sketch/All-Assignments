import numpy as np
Range = np.arange(3, 3)
range = np.arange(0, 20, 2)
print("Range =", Range)
print("Range =", range)
print("range of first element = ",range[3])
want = range[1:4]
print("want =", want) 
import numpy as np

matrix = np.arange(1, 13).reshape(3, 4)

print("\n Matrix =")
print(matrix)
print("\n first element of matrix =", matrix[0,0])
want = matrix[1:3, 1:3]
print("want =", want) 