import numpy as np
import scipy as sp
import sympy as sym
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt

R=1                    
x= sym.Symbol('x')
r= sym.Symbol('r')
ψa_sym=sym.exp(-sym.sqrt(x**2+r**2))/sym.sqrt(sym.pi)
ψb_sym=sym.exp(-sym.sqrt((x-R)**2+r**2))/sym.sqrt(sym.pi)

def Laplacian(f):
    return sym.diff(sym.diff(f, x), x) + sym.diff(r*sym.diff(f, r), r)/r

ψ=(sym.lambdify([x, r], ψa_sym, "numpy"), sym.lambdify([x, r], ψb_sym, "numpy"))
Δψ=(sym.lambdify([x, r], Laplacian(ψa_sym), "numpy"), sym.lambdify([x, r], Laplacian(ψb_sym), "numpy"))

H=np.zeros((2, 2))
S=np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        def H_integrand(x, r):
            return 2*np.pi*r*ψ[i](x, r)*(-0.5*Δψ[j](x, r)-((x**2+r**2)**(-0.5)-((x-R)**2+r**2)**(-0.5))*ψ[j](x, r))
        def S_integrand(x, r):
            return 2*np.pi*r*ψ[i](x, r)*ψ[j](x, r)
        H[i][j], err=sp.integrate.nquad(H_integrand, [[-10, +10], [0, +10]])
        S[i][j], err=sp.integrate.nquad(S_integrand, [[-10, +10], [0, +10]])
print(H, S)
        
