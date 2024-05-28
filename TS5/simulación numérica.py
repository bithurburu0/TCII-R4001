# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:29:10 2024

@author: benja
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import TransferFunction

from pytc2.sistemas_lineales import parametrize_sos, pzmap, GroupDelay, bodePlot, tfcascade
from pytc2.general import print_subtitle

# módulo de análisis simbólico
import sympy as sp
# variable de Laplace
from sympy.abc import s
from IPython.display import display, Math, Markdown


num = [15]
den = [1, 6, 15, 15]
# Se obtienen las raices del denominador de T3(s)
roots = np.roots(den)

alpha1 = np.absolute(np.real(roots[0]))
beta1 = np.imag(roots[0])

alpha2 = np.absolute(np.real(roots[1]))
beta2 = np.imag(roots[1])

alpha3 = np.absolute(np.real(roots[2]))
beta3 = np.imag(roots[2])



#%%
w = np.arange(0.0, 100, 0.01)
D = alpha1/(alpha1**2 + (w + beta1)**2) + alpha2/(alpha2**2 + (w + beta2)**2) + alpha3/(alpha3**2 + (w + beta3)**2)

plt.figure()
plt.semilogx(w, D)
plt.grid()
plt.title('Normalizado')
plt.show()

# Cálculo de error o desviamiento de D(2.5) respecto de D(0)
result = []
for w in [0, 2.5]:
    D = alpha1/(alpha1**2 + (w + beta1)**2) + alpha2/(alpha2**2 + (w + beta2)**2) + alpha3/(alpha3**2 + (w + beta3)**2)
    print(f"D({w})={D}",end="\n")
    result.append(D)

error = (result[0] - result[1])*100
print(error)
#########################################################################
#%%
w, T = signal.freqs(num, den)
plt.figure()
plt.semilogx(w[1:], -np.diff(np.unwrap(np.angle(T)))/np.diff(w))
plt.grid()
plt.title('subplot 2')
plt.show()
#########################################################################
#%%
b, a = signal.bessel(3, 1, 'low', analog=True, norm='delay')
w, h = signal.freqs(b, a)

plt.figure()
plt.semilogx(w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w))
plt.grid()
plt.title('subplot 3')
plt.show()

#########################################################################
#%%

# Simulación simbólica
## Definición de variables simbólicas
s = sp.symbols('s', complex=True)
G1, G2, C1, C2 = sp.symbols("G1, G2, C1, C2")
Vi, Vo, Va, Vb = sp.symbols("Vi, Vo, Va, Vb")

# Sistemas de ecuaciones del modelo Sallen-Key
aa = sp.solve([
                (G1 + G2 + s*C2)*Va - G1*Vi - G2*Vb - s*C2*Vo, 
                (G2 + s*C1)*Vb - G2*Va,
                Vb - Vo
                ], 
                [Vi, Vo, Va, Vb])
T = aa[Vo]/aa[Vi]

num, den = sp.fraction(sp.simplify(sp.expand(T)))
num = sp.Poly(num,s)
den = sp.Poly(den,s)

print_subtitle('Transferencia de Sallen-Key con K=1')

display(Math( r' \frac{V_o}{V_i} = ' + sp.latex(sp.Mul(num/den, evaluate=False)) ))


#########################################################################
#%%

# Valores normalizados de los componentes luego de la sintetilazación
C1_n = alpha1/(alpha1**2+beta1**2)
C2_n = 1/alpha1
C3_n = (alpha1**2 + beta1**2)/15
R1_n = 1
R2_n = 1
R3_n = 1

w = 1
D = alpha1/(alpha1**2 + (w + beta1)**2) + alpha2/(alpha2**2 + (w + beta2)**2) + alpha3/(alpha3**2 + (w + beta3)**2)

# Valores desnormalizados de los componentes, para D(1)=200uSeg
omega_f = 1/(1*(200e-6)/D) # Se calcula D(0) a partir de conocer D(1) -> Regla de 3 simple
C1 = C1_n/omega_f
C2 = C2_n/omega_f
C3 = C3_n/omega_f
R1 = R1_n
R2 = R2_n
R3 = R3_n

print("C1 = ", C1)
print("C2 = ", C2)
print("C3 = ", C3)
print("R1 = ", R1)
print("R2 = ", R2)
print("R3 = ", R3)

#########################################################################
#%%

# Se obtienen los valores de los componentes en función de los parámetros wo y Q

Wo = np.sqrt(alpha1**2 + beta1**2)
Q = Wo/(2*alpha1)
# Valores normalizados de los componentes luego de la sintetilazación
C1_n = 1/(2*Q*Wo)
C2_n = 2*Q/Wo
C3_n = (Wo**2)/15
R1_n = 1
R2_n = 1
R3_n = 1

w = 1
D = alpha1/(alpha1**2 + (w + beta1)**2) + alpha2/(alpha2**2 + (w + beta2)**2) + alpha3/(alpha3**2 + (w + beta3)**2)

omega_f = 1/(1*(200e-6)/D) # Se calcula D(0) a partir de conocer D(1) -> Regla de 3 simple
C1 = C1_n/omega_f
C2 = C2_n/omega_f
C3 = C3_n/omega_f
R1 = R1_n
R2 = R2_n
R3 = R3_n

print("C1 = ", C1)
print("C2 = ", C2)
print("C3 = ", C3)
print("R1 = ", R1)
print("R2 = ", R2)
print("R3 = ", R3)

#########################################################################
#%%

num = [1/(R1*R2*R3)]
den = [C1*C2*C3, (C1*C2)/R3 + (C1*C3)/R1 + (C1*C3)/R2, C1/(R1*R3) + C1/(R2*R3) + C3/(R1*R2), 1/(R1*R2*R3)]

my_tf = TransferFunction(num, den)
fig1, ax = GroupDelay(my_tf)


