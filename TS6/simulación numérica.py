# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:05:35 2024

@author: benja
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import TransferFunction

# módulo de análisis simbólico
import sympy as sp
# variable de Laplace
from sympy.abc import s
from IPython.display import display, Math, Markdown

from pytc2.sistemas_lineales import parametrize_sos, pzmap, GroupDelay, bodePlot, tfcascade, tf2sos_analog
from pytc2.general import print_subtitle

#%%
# alpha_min = 16 //valor requerido
alpha_max = 0.5
ee_2 = 10**(alpha_max/10)-1
ee = np.sqrt(ee_2)

Q = 5
fs1 = 17e3
f0 = 22e3
w0 = 2*np.pi*f0
ws1 = 2*np.pi*fs1
ws1_n = ws1/w0
Wp_n = 1
Ws_n = np.abs(Q*(ws1_n**2-1)/ws1_n)

# Se itera n hasta que alpha sea mayor a alpha_min
n=3
alpha_min = 10*np.log10(1+ee_2*np.cosh(n*np.arccosh(Ws_n))**2)

#%%

# Se obtienen las raices del denominador de |T(s)|**2
roots = np.roots([ 1 , 0 , 3/2 , 0 , 9/16 , 0 , -1/(16*ee_2) ])

# Se obtienen los coeficientes de las transferencias a partir de las raices
# T1(s)
alpha1 = np.absolute(np.real(roots[4]))

# T2(s)
alpha2 = np.absolute(np.real(roots[0]))
beta2 = np.imag(roots[0])

2*alpha2 
alpha2**2 + beta2**2


#%%

# Simulación simbólica
## Definición de variables simbólicas
s = sp.symbols('s', complex=True)
a1, a2 = sp.symbols("a1, a2")
b2 = sp.symbols("b2")
epsilon = sp.symbols("e")

## Obtención de transferencia en función de alpha y beta
T1 = -sp.sqrt(1/(16*epsilon**2))/(s + a1)
T2 = 1/(s**2 + 2*a2*s + (a2**2 + b2**2))

T = T1*T2

num1, den1 = sp.fraction(sp.simplify(sp.expand(T1)))
num1 = sp.Poly(num1,s)
den1 = sp.Poly(den1,s)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num1/den1)) )

num2, den2 = sp.fraction(sp.simplify(sp.expand(T2)))
num2 = sp.Poly(num2,s)
den2 = sp.Poly(den2,s)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num2/den2)) )

num, den = sp.fraction(sp.simplify(sp.expand(T)))
num = sp.Poly(num,s)
den = sp.Poly(den,s)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num/den)) )
#%%

p = sp.symbols('p', complex=True)
Q = sp.symbols("Q")
s = Q*(p**2+1)/p

T1 = -sp.sqrt(1/(16*epsilon**2))/(s + a1)
T2 = 1/(s**2 + 2*a2*s + (a2**2 + b2**2))

T = T1*T2

num1, den1 = sp.fraction(sp.simplify(sp.expand(T1)))
num1 = sp.Poly(num1,p)
den1 = sp.Poly(den1,p)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num1/den1)) )

num2, den2 = sp.fraction(sp.simplify(sp.expand(T2)))
num2 = sp.Poly(num2,p)
den2 = sp.Poly(den2,p)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num2/den2)) )

num, den = sp.fraction(sp.simplify(sp.expand(T)))
num = sp.Poly(num,p)
den = sp.Poly(den,p)
print_subtitle('Transferencia obtenida en función de alpha y beta')
display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(num/den)) )

#%%
pytc2.sistemas_lineales.tf2sos_analog(num, den=[])
