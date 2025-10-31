```
import numpy as np
import sympy as sp
```

```
# transpose of vector matrix
NR=int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
entries= list(map(float,input().split()))
A=np.array(entries).reshape(NR,NC)
Transpose= np.transpose(A)

print('transpose of matrix A:-', Transpose)
```

```
# ECHELON FORM AND RANK OF A MATRIX

NR= int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
elements=[]
print('enter elements row by row:-')
for i in range(NR):
  NR=list(map(float,input().split()))
  elements.append(NR)
A= np.array(elements)
A=Matrix(A)
print('user defdined matrix:-')

echelon= A.echelon_form()
rank= A.rank()
B=Matrix(echelon)

print('echelon form :-',B)
print('rank:-', rank)
```

```                                 # FINDING ADJOINT, INVERSE, COFACTOR AND TRANSPOSE OF MATRIX
NR= int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
elements=[]
print('enter elements row by row:-')
for i in range(NR):
  row=list(map(float,input().split())) # Use float to avoid type issues later
  elements.append(row)
A_sympy= Matrix(elements) # Keep the SymPy matrix for potential future use
A_numpy = np.array(elements, dtype=np.float64) # Convert to NumPy array with float type
print('user defined matrix (SymPy):-', A_sympy)
print('user defined matrix (NumPy):-', A_numpy)

try:
    DETERMINANT = np.linalg.det(A_numpy)
    print('\ndeterminant of matrix:-', DETERMINANT)

    if DETERMINANT != 0:
        INVERSE = np.linalg.inv(A_numpy)
        print('\ninverse of matrix:-', INVERSE)

        TRANSPOSE = np.transpose(A_numpy)
        print('\ntranspose of matrix:-', TRANSPOSE)

        # Calculate the cofactor and adjoint matrices
        # The cofactor matrix is the transpose of the adjoint matrix
        # adj(A) = det(A) * inv(A)
        ADJOINT = DETERMINANT * INVERSE
        COFACTOR = np.transpose(ADJOINT)

        print('\ncofactor of matrix:-', COFACTOR)
        print('\nadjoint of matrix:-', ADJOINT)

    else:
        print("\nMatrix is singular, inverse does not exist.")

except np.linalg.LinAlgError:
    print("\nCould not calculate inverse. Matrix might be singular or not square.")
```


```
# solving homogeneous system of equation using gauss elimination

NR= int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
elements=[]
print('enter elements row by row:-')
for i in range(NR):
  row=list(map(float,input().split())) # Use float to avoid type issues later
  elements.append(row)
A_sympy= Matrix(elements) # Keep the SymPy matrix for potential future use
A_numpy = np.array(elements, dtype=np.float64) # Convert to NumPy array with float type
print('user defined matrix (SymPy):-', A_sympy)
print('user defined matrix (NumPy):-', A_numpy)

Constant_Matrix= np.zeros(NR)
X= np.linalg.solve(A_numpy, Constant_Matrix)
print('UNIQUE SOLUTION:-', X)
```

```
# SOLVING homogeneous system using gauss jordan

NR= int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
elements=[]
print('enter elements row by row:-')
for i in range(NR):
  row=list(map(float,input().split())) # Use float to avoid type issues later
  elements.append(row)
A_sympy= Matrix(elements) # Keep the SymPy matrix for potential future use
A_numpy = np.array(elements, dtype=np.float64) # Convert to NumPy array with float type
print('user defined matrix (SymPy):-', A_sympy)
print('user defined matrix (NumPy):-', A_numpy)

column_entries= list(map(float,input().split()))
column_matrix= np.array(column_entries).reshape(NR,1)

print('coeffiient matrix A:-', '\n', A_numpy)
print('column matrix B:-', '\n', column_matrix)

INVERSE_A= np.linalg.inv(A_numpy)
solution= np.matmul(INVERSE_A, column_matrix)
print(solution)
```


```
