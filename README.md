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
A=sp.Matrix(A)
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
A_sympy= sp.Matrix(elements) 
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
A_sympy= sp.Matrix(elements) 
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
A_sympy= sp.Matrix(elements) 
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
A_sympy= sp.Matrix(elements) 
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

m_columnspace= sp.Matrix(A).columnspace()
m_rowspace= sp.Matrix(A).rowspace()

print('columnspace of matrix A:-', m_columnspace)
print('rowspace of matrix A:-', m_rowspace)
```


```
# checking linear dependence of vectors and generate a linear combination of given vectors of matrices

NR= int(input("Enter the number of rows:"))
NC= int(input("Enter the number of columns:"))
print("Enter the entries in a single line (seperated by space)")
entries = list (map(int, input().split()))
A=np.array(entries).reshape(NR,NC)
A=sp.Matrix(A)
rank = A.rank()
print(f"\nRank = {rank}")
if rank == NC:
    print("Linearly independent.")
else:
    print("Linearly dependent.")
    ns = A.nullspace()
    for v in ns:
        expr = " + ".join([f"({v[i]})v{i+1}" for i in range(NC)]) + " = 0"
        print(expr)
```


```
# Finding the orthonormal basis of given vector space using gram-schmidt orthogonalization process


n = int(input("Enter vector dimension: "))
print("Enter 3 vectors (each with", n, "elements):")
v = [np.array(list(map(float, input().split()))) for _ in range(3)]

# Gram-Schmidt process
u = [v[0]]
u.append(v[1] - np.dot(v[1], u[0]) / np.dot(u[0], u[0]) * u[0])
u.append(v[2] - sum(np.dot(v[2], ui) / np.dot(ui, ui) * ui for ui in u[:2]))

# Normalize
e = [ui / np.linalg.norm(ui) for ui in u]

# Output
print("\nOrthonormal basis vectors:")
for i, ei in enumerate(e, 1):
    print(f"e{i} =", np.round(ei, 4))
```


```
# checking the diagonizable property of matrices and finding the corresponding wife values and verify cayley Hamilton theorem

import numpy as np
import sympy as sp

NR= int(input("Enter the number of rows:"))
NC= int(input("Enter the number of columns:"))
print("Enter the entries in a single line (seperated by space)")
entries = list (map(int, input().split()))
A=np.array(entries).reshape(NR,NC)

M = sp.Matrix(A)
try:
    P, D = M.diagonalize()
    print("Matrix is diagonalizable.")
    print("\nP (Eigenvectors):\n", P)
    print("\nD (Diagonal Matrix of Eigenvalues):\n", D)
except:
    print("Matrix is not diagonalizable.")

    p = M.charpoly()
    print("\nCharacteristic Polynomial:", p.as_expr())
    print("Verifying Cayley-Hamilton Theorem...")
    cayley_hamilton_result = p.eval(M)
    print("p(A) =\n", cayley_hamilton_result)
    if cayley_hamilton_result == sp.zeros(*M.shape):
        print("Cayley-Hamilton theorem verified successfully!")
    else:
        print("Cayley-Hamilton theorem not satisfied (possible rounding or symbolic issue).")
```


```
# Linear algebra :- Coding and Decoding of message using non singular matrices.

import math
from sympy import Matrix

msg = "LINEAR algebra is FUN"
# prepare numeric vector: A=1..Z=26, other -> 0 (space/pad)
nums = [ord(c.upper())-64 if c.isalpha() else 0 for c in msg]
# pad to multiple of 3
pad = (-len(nums)) % 3
nums += [0]*pad

K = Matrix([[2,3,1],[1,1,1],[1,2,2]])
Kinv = K.inv()

# break into blocks of 3 and encode each block (no reshape)
blocks = [Matrix(nums[i:i+3]) for i in range(0, len(nums), 3)]
encoded_blocks = [K * b for b in blocks]
encoded = [int(x) for B in encoded_blocks for x in list(B)]
print("Encoded:", encoded)

# decode blockwise and reassemble
decoded_blocks = [Kinv * B for B in encoded_blocks]
decoded_nums = [round(int(x)) if float(x).is_integer() else round(float(x)) 
                for D in decoded_blocks for x in list(D)]
decoded_text = ''.join(chr(n+64) if 1 <= n <= 26 else ' ' for n in decoded_nums).strip()
print("Decoded:", decoded_text)
```


```
# gradient of vector field

x, y, z = sp.symbols('x y z')
expr_input = input("Enter scalar function f(x,y,z) [default: x**2*y + sin(z)]: ").strip()
if expr_input == "":
    expr_input = "x**2*y + sin(z)"
f = sp.sympify(expr_input)
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
df_dz = sp.diff(f, z)
print("\nFunction f(x,y,z) =", f)
print("Gradient ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]")
print("∂f/∂x =", df_dx)
print("∂f/∂y =", df_dy)
print("∂f/∂z =", df_dz)
```



```
# divergence of vector field

x, y, z = sp.symbols('x y z')
P = input("Enter P(x,y,z) [default: x*y]: ").strip()
Q = input("Enter Q(x,y,z) [default: y*z]: ").strip()
R = input("Enter R(x,y,z) [default: z*x]: ").strip()

if P == "": P = "x*y"
if Q == "": Q = "y*z"
if R == "": R = "z*x"

P = sp.sympify(P)
Q = sp.sympify(Q)
R = sp.sympify(R)
div = sp.diff(P, x) + sp.diff(Q, y) + sp.diff(R, z)
print("\nVector Field F =", (P, Q, R))
print("Divergence ∇·F =", sp.simplify(div))
```



```
# curl of a vector field

x, y, z = sp.symbols('x y z')

P = input("Enter P(x,y,z) [default: y*z]: ").strip()
Q = input("Enter Q(x,y,z) [default: z*x]: ").strip()
R = input("Enter R(x,y,z) [default: x*y]: ").strip()

if P == "": P = "y*z"
if Q == "": Q = "z*x"
if R == "": R = "x*y"

P = sp.sympify(P)
Q = sp.sympify(Q)
R = sp.sympify(R)

curl_x = sp.diff(R, y) - sp.diff(Q, z)
curl_y = sp.diff(P, z) - sp.diff(R, x)
curl_z = sp.diff(Q, x) - sp.diff(P, y)

print("\nVector Field F =", (P, Q, R))
print("Curl ∇×F = [∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y]")
print("∂R/∂y - ∂Q/∂z =", curl_x)
print("∂P/∂z - ∂R/∂x =", curl_y)
print("∂Q/∂x - ∂P/∂y =", curl_z)
```


