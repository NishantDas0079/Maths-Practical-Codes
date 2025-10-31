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
