import numpy as np
import scipy as sp
import sympy as sym
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt

R=1                    
x= sym.Symbol('x')
y= sym.Symbol('y')
z= sym.Symbol('z')
ψa_sym=sym.exp(-sym.sqrt(x**2+y**2+z**2))/sym.sqrt(sym.pi)
ψb_sym=sym.exp(-sym.sqrt((x-R)**2+y**2+z**2))/sym.sqrt(sym.pi)

def Laplacian(f):
    return sym.diff(sym.diff(f, x), x)+sym.diff(sym.diff(f, y), y)+sym.diff(sym.diff(f, z), z)

ψa=sym.lambdify([x, y, z], ψa_sym, "numpy")
ψb=sym.lambdify([x, y, z], ψb_sym, "numpy")
Δψa=sym.lambdify([x, y, z], Laplacian(ψa_sym), "numpy")
Δψb=sym.lambdify([x, y, z], Laplacian(ψb_sym), "numpy")

H=np.zeros((2, 2))
S=np.zeros((2, 2))