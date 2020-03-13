import numpy as np
import scipy as sp
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
             
N=40
nMax=6
radiusMax=40
α=[0.001 * (1000000)**(i/N) for i in range(N)]  # α runs from 0.001 to 1000 (geometric progression) 
H=np.zeros((N, N))
S=np.zeros((N, N)) # H (hamiltonian) and S (overlap) are just two square matrices (I initialized them with zero, but this value is overwritten in the following loop)

for i in range(N):
        for j in range(N):
                H[i][j]=0.5*6*α[i]*α[j]*(np.pi/(α[i]+α[j]))**1.5/(α[i]+α[j])-2*np.pi/(α[i]+α[j])
                S[i][j]=(np.pi/(α[i]+α[j]))**1.5
                
alpha, beta, vl, C, work, info = sp.linalg.lapack.zggev(H, S) # zggev is the function to perform the generalized eigenvalue problem
E=(alpha/beta).real #the eigenvalues are actually real, but i casted them in real to avoid tedious warnings

def sortIndex(array): #sort a list, but modify only the list of the indices
    a=list(array)
    L=list(range(len(a)))
    repeat=True
    while repeat:
        repeat=False
        for i in range(len(a)-1):
            if a[i]>a[i+1]:
                a[i], a[i+1]=a[i+1], a[i]
                L[i], L[i+1]=L[i+1], L[i]
                repeat=True
    return L
L=sortIndex(E) # I cannot change the original eigenvalue array, so I decided to reroll a "index" array

def eigenfunction(x, index):
    y=0
    for i in range(N):
        y+=(C[:,index][i].real)*np.e**(-α[i]*x**2) # C[:, j] is the array of the coefficients (eigenvector)
    return y
def radialProbability(x, index):
    return x**2*eigenfunction(x, index)**2
index=0
for j in L[0:nMax]:
    index+=1
    def toeV(E):
        return 27.211386245988*E
    print("Energy eigenvalue n=%d : %.8f (%.8f eV)  ...  expected : %.8f (%.8f eV)" % (index, E[j], toeV(E[j]), -0.5/(index**2), toeV(-0.5/(index**2))))
    xvalues = np.linspace(0, radiusMax, 1000)
    yvalues = []
    I, err=sp.integrate.quad(radialProbability, 0, np.inf, j)# I is the normalization constant (quad is the function to compute a definite integral)
    for x in xvalues:
        y=radialProbability(x, j)/I
        yvalues.append(y)
    plt.plot(xvalues, yvalues)
bottom, top = plt.ylim()
plt.xlim(0, radiusMax)
plt.ylim(0, top)
plt.xlabel('Radius ($a_0$)')
plt.ylabel('$r^2[R(r)]^2$ (ad.)')
plt.show()