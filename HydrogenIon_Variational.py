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

ψa=sym.lambdify([x, r], ψa_sym, "numpy")
ψb=sym.lambdify([x, r], ψb_sym, "numpy")
Δψa=sym.lambdify([x, r], Laplacian(ψa_sym), "numpy")
Δψb=sym.lambdify([x, r, Laplacian(ψb_sym), "numpy")

H=np.zeros((2, 2))
S=np.zeros((2, 2))
