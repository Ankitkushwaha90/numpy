Here is a comprehensive collection of NumPy code examples, covering a wide range of topics, from basic array manipulations to more advanced operations like broadcasting, linear algebra, and random sampling.

### 1. Creating Arrays
```python
import numpy as np

# Create a 1D array
a = np.array([1, 2, 3])
print("1D array:", a)

# Create a 2D array
b = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:\n", b)

# Create an array of zeros
zeros = np.zeros((3, 3))
print("Array of zeros:\n", zeros)

# Create an array of ones
ones = np.ones((2, 4))
print("Array of ones:\n", ones)

# Create an array with a range of numbers
range_array = np.arange(0, 10, 2)
print("Range array:", range_array)

# Create an array with evenly spaced numbers between two values
linspace_array = np.linspace(0, 1, 5)
print("Linspace array:", linspace_array)
```
### 2. Basic Array Operations
```python
import numpy as np

# Element-wise addition, subtraction, multiplication, and division
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

add_result = a + b
sub_result = a - b
mul_result = a * b
div_result = a / b

print("Addition:", add_result)
print("Subtraction:", sub_result)
print("Multiplication:", mul_result)
print("Division:", div_result)

# Element-wise square root
sqrt_result = np.sqrt(a)
print("Square root:", sqrt_result)

# Sum, mean, max, and min
array = np.array([[1, 2, 3], [4, 5, 6]])
print("Sum:", np.sum(array))
print("Mean:", np.mean(array))
print("Max:", np.max(array))
print("Min:", np.min(array))
```
### 3. Reshaping and Transposing Arrays
```python
import numpy as np

# Reshape a 1D array to a 2D array
array = np.arange(6)
reshaped = array.reshape((2, 3))
print("Reshaped array:\n", reshaped)

# Transpose a matrix (swap rows and columns)
array = np.array([[1, 2, 3], [4, 5, 6]])
transposed = np.transpose(array)
print("Transposed array:\n", transposed)

# Flatten a multi-dimensional array to 1D
flattened = array.flatten()
print("Flattened array:", flattened)
```
### 4. Indexing and Slicing
```python
import numpy as np

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing specific elements
print("Element at (1, 2):", array[1, 2])  # Access element at row 1, column 2

# Slicing rows and columns
print("First row:", array[0, :])
print("Second column:", array[:, 1])

# Slice a sub-array
sub_array = array[0:2, 1:3]
print("Sub-array:\n", sub_array)
```
### 5. Broadcasting
```python
import numpy as np

# Broadcasting a scalar to an array
array = np.array([1, 2, 3])
scalar = 2
broadcast_result = array * scalar
print("Broadcasting scalar to array:", broadcast_result)

# Broadcasting two arrays with compatible shapes
a = np.array([[1], [2], [3]])  # Shape (3, 1)
b = np.array([4, 5, 6])        # Shape (1, 3)
broadcast_sum = a + b
print("Broadcast result:\n", broadcast_sum)
```
### 6. Random Sampling
```python
import numpy as np

# Generate random numbers between 0 and 1
random_array = np.random.rand(3, 3)
print("Random array:\n", random_array)

# Generate random integers between two values
random_integers = np.random.randint(0, 10, size=(3, 3))
print("Random integers:\n", random_integers)

# Generate random normal distribution (mean=0, std=1)
random_normal = np.random.randn(3, 3)
print("Random normal distribution:\n", random_normal)

# Shuffle an array
array = np.array([1, 2, 3, 4, 5])
np.random.shuffle(array)
print("Shuffled array:", array)
```
### 7. Linear Algebra Operations
```python
Copy code
import numpy as np

# Create two matrices
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication (dot product)
dot_product = np.dot(matrix_a, matrix_b)
print("Dot product:\n", dot_product)

# Matrix determinant
determinant = np.linalg.det(matrix_a)
print("Determinant:", determinant)

# Inverse of a matrix
inverse = np.linalg.inv(matrix_a)
print("Inverse matrix:\n", inverse)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix_a)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```
### 8. Array Aggregation and Sorting
```python
import numpy as np

array = np.array([[1, 4, 3], [9, 2, 8]])

# Sum, mean, max, min along different axes
print("Sum (axis=0):", np.sum(array, axis=0))  # Column-wise sum
print("Sum (axis=1):", np.sum(array, axis=1))  # Row-wise sum
print("Mean (axis=0):", np.mean(array, axis=0))
print("Max (axis=0):", np.max(array, axis=0))
print("Min (axis=1):", np.min(array, axis=1))

# Sort array along rows
sorted_array = np.sort(array, axis=1)
print("Sorted array:\n", sorted_array)
```
### 9. Stacking and Splitting Arrays
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Horizontal stacking (along columns)
h_stack = np.hstack((a, b))
print("Horizontal stack:\n", h_stack)

# Vertical stacking (along rows)
v_stack = np.vstack((a, b))
print("Vertical stack:\n", v_stack)

# Splitting arrays
split_array = np.hsplit(h_stack, 2)
print("Split arrays:\n", split_array)
```
### 10. Mathematical Functions
```python
import numpy as np

# Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
sin_values = np.sin(angles)
cos_values = np.cos(angles)
print("Sin values:", sin_values)
print("Cos values:", cos_values)

# Exponential and logarithmic functions
exponent = np.exp([1, 2, 3])
logarithm = np.log([1, np.e, np.e**2])
print("Exponential:", exponent)
print("Logarithm:", logarithm)

# Rounding functions
arr = np.array([1.2, 3.5, 4.7])
print("Floor:", np.floor(arr))
print("Ceil:", np.ceil(arr))
print("Round:", np.round(arr))
```
### 11. Boolean Indexing and Filtering
```python
import numpy as np

array = np.array([1, 2, 3, 4, 5, 6])

# Boolean indexing (filtering based on condition)
filtered_array = array[array > 3]
print("Filtered array:", filtered_array)

# Boolean mask for 2D arrays
matrix = np.array([[1, 2], [3, 4], [5, 6]])
mask = matrix > 3
print("Mask:\n", mask)
print("Elements greater than 3:\n", matrix[mask])
```
### 12. Unique Elements and Histogram
```python
import numpy as np

array = np.array([1, 2, 3, 1, 2, 4, 5, 5, 6])

# Unique elements
unique_elements = np.unique(array)
print("Unique elements:", unique_elements)

# Histogram
hist, bins = np.histogram(array, bins=3)
print("Histogram:", hist)
print("Bin edges:", bins)
```
These examples showcase the wide range of capabilities in NumPy, from basic array manipulation and mathematical operations to more advanced topics like linear algebra, broadcasting, and random sampling. Each example is designed to be intuitive and easy to understand for learners at all levels.
