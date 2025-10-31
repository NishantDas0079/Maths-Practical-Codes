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
# finding null space and nullity of matrix
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

nullspace = A_sympy.nullspace()
# nullspace=Matrix() # This line seems unnecessary, so I've commented it out

NoC= A_numpy.shape[1]
rank = A_sympy.rank()
nullity= NoC-rank
print('nullity of matrix A:-', nullity)
print('null space of matrix A:-', nullspace)
```

```
# finding columnspace and rowspace

NR=int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
entries= list(map(float,input().split()))
A=np.array(entries).reshape(NR,NC)

m_columnspace= Matrix(A).columnspace()
m_rowspace= Matrix(A).rowspace()

print('columnspace of matrix A:-', m_columnspace)
print('rowspace of matrix A:-', m_rowspace)
```


```
# checking linear dependence of vectors and generate a linear combination of given vectors of matrices
NR=int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
entries= list(map(float,input().split()))
A_numpy=np.array(entries).reshape(NR,NC)
A_sympy = Matrix(A_numpy)

n=int(input('enter no. of vectors:-'))
print('enter vectors:-')

rank= A_sympy.rank()
if rank<n:
  print('vectors are linearly dependent')
else:
  print('vectors are linearly independent')

print('linear combination:-')
print('enter coefficients for linear combination:-')

coeffs= list(map(float,input().split()))
result= sum(c*A_sympy.row(i) for i, c in enumerate(coeffs))
print(f'result:- {result}')

if A_sympy.shape[0]==A_sympy.shape[1] and rank==n:
  print('transition matrix:-', A_sympy.inv())
```



```
# Finding the orthonormal basis of given vector space using gram-schmidt orthogonalization process

n=int(input('enter no. of vectors:-'))
print('enter vectors:-')
vectors=[]
dim=int(input('enter dimension of each vector:-'))
for i in range(n):
  print(f'enter vector {i+1} elements separated by space:-')
  vec=list(map(float,input().split()))
  vectors.append(vec)
NR=int(input('enter no. of rows:-'))
NC=int(input('enter no. of columns:-'))
entries= list(map(float,input().split()))
A_numpy=np.array(entries).reshape(NR,NC)
A_sympy=Matrix(A_numpy)

orthogonal=[]
orthonormal=[]
for i in range(n):
  v=A_numpy[i]
  u=v
  for j in range(i):
    proj= v.dot(orthogonal[j]/orthogonal[j].dot(orthogonal[j]))
    u=u-proj
  orthogonal.append(u) # Append the orthogonal vector after computing it
  norm= sqrt(u.dot(u))
  orthonormal.append(np.array(simplify(u/norm)).T)
print('orthonormal basis')

for i, entries in enumerate(orthonormal):
       print(f"u{i+1}={entries}")
```


```
