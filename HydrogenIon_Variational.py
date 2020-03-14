import numpy as np
from numpy import pi
import scipy as sp
import sympy as sym
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt

R=2             
x= sym.Symbol('x')
r= sym.Symbol('r')
ψ_sym=(sym.exp(-sym.sqrt((x+R/2)**2+r**2))/sym.sqrt(sym.pi), sym.exp(-sym.sqrt((x-R/2)**2+r**2))/sym.sqrt(sym.pi))

def Laplacian(f):
    return sym.diff(sym.diff(f, x), x) + sym.diff(r*sym.diff(f, r), r)/r

ψ=(sym.lambdify([x, r], ψ_sym[0], "numpy"), sym.lambdify([x, r], ψ_sym[1], "numpy"))
Laplψ=(sym.lambdify([x, r], Laplacian(ψ_sym[0]), "numpy"), sym.lambdify([x, r], Laplacian(ψ_sym[1]), "numpy"))

"""
H=np.zeros((2, 2))
S=np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        def H_integrand(x, r):
            return 2*pi*r*ψ[i](x, r)*(-0.5*Laplψ[j](x, r)-(((x+R/2)**2+r**2)**(-0.5)+((x-R/2)**2+r**2)**(-0.5))*ψ[j](x, r))
        def S_integrand(x, r):
            return 2*pi*r*ψ[i](x, r)*ψ[j](x, r)
        H[i][j], err=sp.integrate.nquad(H_integrand, [[-np.inf, +np.inf], [0, +np.inf]])
        S[i][j], err=sp.integrate.nquad(S_integrand, [[-np.inf, +np.inf], [0, +np.inf]])     
"""

def H11_integrand(x, r):
    return 2*pi*r*ψ[0](x, r)*(-0.5*Laplψ[0](x, r)-(((x+R/2)**2+r**2)**(-0.5)+((x-R/2)**2+r**2)**(-0.5))*ψ[0](x, r))
def H22_integrand(x, r):
    return 2*pi*r*ψ[0](x, r)*(-0.5*Laplψ[1](x, r)-(((x+R/2)**2+r**2)**(-0.5)+((x-R/2)**2+r**2)**(-0.5))*ψ[1](x, r))
def S_integrand(x, r):
    return 2*pi*r*ψ[0](x, r)*ψ[1](x, r)
H11, err=sp.integrate.nquad(H11_integrand, [[-np.inf, +np.inf], [0, +np.inf]])
H22, err=sp.integrate.nquad(H22_integrand, [[-np.inf, +np.inf], [0, +np.inf]])
S, err=sp.integrate.nquad(S_integrand, [[-np.inf, +np.inf], [0, +np.inf]])

print((H11-H22)/(1-S))
print((H11+H22)/(1+S))
