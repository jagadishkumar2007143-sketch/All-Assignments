import numpy as np

# Input two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 1. Basic Operations
print("Vector 1 =", v1)
print("Vector 2 =", v2)

print("Addition =", v1 + v2)
print("Subtraction =", v1 - v2)
print("Dot Product =", np.dot(v1, v2))

# 2. Closest Vector
vectors = np.array([
    [2, 3, 4],
    [5, 5, 5],
    [1, 2, 2]
])

target = np.array([1, 2, 3])

distance = np.linalg.norm(vectors - target, axis=1)

index = np.argmin(distance)

print("\nTarget Vector =", target)
print("Closest Vector =", vectors[index])

# 3. Magnitude and Direction (Unit Vector)
magnitude = np.linalg.norm(v1)
direction = v1 / magnitude

print("\nMagnitude =", magnitude)
print("Direction (Unit Vector) =", direction)