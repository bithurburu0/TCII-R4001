import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import TransferFunction

from pytc2.sistemas_lineales import simplify_n_monic, parametrize_sos, pzmap, GroupDelay, bodePlot, tfcascade
from pytc2.general import print_subtitle

# módulo de análisis simbólico
import sympy as sp
# variable de Laplace
from sympy.abc import s
from IPython.display import display, Math, Markdown


alfa_max = 1
ee_2 = 10**(alfa_max/10)-1
ee = np.sqrt(ee_2)

# Código para análisis del filtro de orden 3 obtenido
num = [1/ee]
den = [1, 2/np.cbrt(ee), 2/np.cbrt(ee_2), 1/ee]

# Obtención de función transferencia
my_tf = TransferFunction(num,den)

# Gráfico Bode
bodePlot(my_tf, fig_id=1)

# Gráfico de Polos y Ceros
pzmap(my_tf, True, fig_id=2)

# Group Delay
GroupDelay(my_tf, fig_id=3)