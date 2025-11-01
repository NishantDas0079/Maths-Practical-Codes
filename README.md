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
A_sympy=sp.Matrix(A_numpy)

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

mod = 257
NR= int(input("Enter the number of rows:"))
NC= int(input("Enter the number of columns:"))
print("Enter the entries in a single line (seperated by space)")
entries = list (map(int, input().split()))
A=np.array(entries).reshape(NR,NC)

def encode(msg):
    v = [ord(c) for c in msg]
    while len(v) % 3: v.append(0)
    res = []
    for i in range(0, len(v), 3):
        b = np.array(v[i:i+3])
        c = (A @ b) % mod
        res += list(c)
    return res

def decode(codes):
    Kinv = np.array(Matrix(A).inv_mod(mod)).astype(int)
    res = []
    for i in range(0, len(codes), 3):
        b = np.array(codes[i:i+3])
        p = (Kinv @ b) % mod
        res += list(p)
    return ''.join(chr(x) for x in res if x != 0)

msg = "Linear algebra is fun"
enc = encode(msg)
dec = decode(enc)
print("Encoded:", enc)
print("Decoded:", dec)
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


