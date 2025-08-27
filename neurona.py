#@title import packages
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors
import matplotlib as mpl
from scipy.signal import find_peaks
from matplotlib.pyplot import cm
from matplotlib import cm
from matplotlib import gridspec
from scipy.integrate import odeint
from numpy.linalg import *
import time
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from tempfile import TemporaryFile
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class neurona:
    def __init__(self):
        # parámetros biofísicos
        self.C = 1;  #Representa la capacidad de la membrana para almacenar carga, como un condensador. Un valor de 1 μF/cm² es típico en neuronas. Determina qué tan rápido responde el voltaje a las corrientes, afectando la forma del spike.
        self.El = -65; #Es el potencial de equilibrio de la corriente de fuga, en mV. Mantiene la neurona en un estado estable cuando no hay estímulos.
        self.Ena = 55; #Potencial de equilibrio del sodio, en mV. La diferencia  𝑣−𝐸_𝑁𝑎  impulsa la entrada de 𝑁𝑎+, iniciando el spike.
        self.Ek = -75; #Potencial de equilibrio del potasio, en mV.  La diferencia 𝑣−𝐸_𝑘  impulsa la salida de 𝐾+, repolarizando la membrana tras el spike.
        self.Gl = 0.025; #Conductancia de la corriente de fuga, en mS/cm². Estabiliza el voltaje en reposo, contrarrestando corrientes excitatorias.
        self.Gna = 120; # Conductancia máxima de los canales de sodio, en mS/cm². Es clave para generar el ascenso rápido del spike.
        self.Gk = 36; # Conductancia máxima de los canales de potasio, en mS/cm². Es esencial para la repolarización (bajada del voltaje) tras el spike.
        self.Iapp = 3.1; #Corriente externa aplicada, en μA/cm². Inicia los potenciales de acción.. #chequear que se tiene q dejar fijo

        self.taurse_i = 0.01;  #Constante de tiempo de subida de la corriente sináptica, en ms. Significado: Indica qué tan rápido se activa la corriente sináptica cuando una neurona presináptica genera un spike. Un valor pequeño (0.3 ms) implica una activación rápida.
        self.taudec_i = 9.0; #Constante de tiempo de decaimiento de la corriente sináptica, en ms. Significado: Indica qué tan rápido decae la corriente sináptica tras la activación. Un valor mayor (9 ms) implica un decaimiento lento.
        self.Ein = -80; #Potencial de equilibrio de la corriente sináptica, en mV. Significado: Es el voltaje al que la corriente sináptica no fluye. Negativo (-80 mV) indica una sinapsis inhibitoria (probablemente GABAérgica).

        self.Gin12 = 0.1; #Conductancia sináptica de la neurona 1 a la neurona 2, en mS/cm². Significado: Mide la fuerza de la conexión sináptica de la neurona 1 (presináptica) a la neurona 2 (postsináptica).
        self.Gin21 = self.Gin12; #Conductancia sináptica de la neurona 2 a la neurona 1, en mS/cm². Significado: idem

        self.alpha = 0.85
    
        #para las ecuaciones:
        self.tappl2 = 8
        self.tconn = 500

    #funciones de activacion/inactivacion
    def minf(self,v): return 1/(1+np.exp(-(v+40)/9))  #Representa la activación de los canales de sodio (𝑚_∞).Es la probabilidad de que las "puertas de activación" estén abiertas, permitiendo que los iones de sodio (𝑁𝑎+) entren y despolaricen la membrana (suban el voltaje).
    def hinf(self,v): return 1/(1+np.exp((v+62)/10)) #Representa la inactivación de los canales de sodio (ℎ∞ ). Es la probabilidad de que las "puertas de inactivación" estén cerradas, bloqueando la entrada de sodio para limitar el potencial de acción.
    def ninf(self,v): return 1/(1+np.exp(-(v+53)/16)) #Calcula la probabilidad de activación de los canales de potasio (𝑛_∞) en equilibrio.
    
    def taum(self,v): return 0.3 + 1e-12*v #Define la constante de tiempo (en ms) para la activación de los canales de sodio (m).
    def tauh(self,v): return 1 + 11/(1+np.exp((v+62)/10)) #Define la constante de tiempo (en ms) para la inactivación de los canales de sodio (h).
    def taun(self,v): return 1 + 6/(1+np.exp((v+53)/16)) #Define la constante de tiempo (en ms) para la activación de los canales de potasio (n).
    
    def H(self,v): return 0.5 * (1 + np.tanh((v + 0) / 4)) #Calcula la probabilidad de activación sináptica basada en el voltaje presináptico. Representa cómo una neurona presináptica activa la sinapsis hacia otra neurona.
    
    #Función de derivadas para el modelo reducido
    def Fa (self, v, n, Iapp, Gl, Gna, Gk): return (Iapp - Gl * (v - self.El) - Gna * self.minf(v)**3 * (self.alpha - n) * (v - self.Ena) - Gk * n**4 * (v - self.Ek)) / self.C

    def network_reduced_deriv(self, y, t):
        V1, n1, Si1, V2, n2, Si2 = y

        # Efecto de la sinapsis solo después de tconn
        Gin12eff = self.Gin12 if t >= self.tconn else 0
        Gin21eff = self.Gin21 if t >= self.tconn else 0

        # Cambio del voltaje de cada neurona usando Fa y la corriente sináptica
        dV1dt = (self.Fa(V1, n1, self.Iapp, self.Gl, self.Gna, self.Gk) - Gin12eff * Si2 * (V1 - self.Ein)) / self.C
        dV2dt = (self.Fa(V2, n2, self.Iapp, self.Gl, self.Gna, self.Gk) - Gin21eff * Si1 * (V2 - self.Ein)) / self.C

        # Cambio de la variable de potasio
        dn1dt = (self.ninf(V1) - n1) / self.taun(V1)
        dn2dt = (self.ninf(V2) - n2) / self.taun(V2)

        # Cambio de la variable sináptica
        dSi1dt = self.H(V1) * (1 - Si1) / self.taurse_i - Si1 / self.taudec_i
        dSi2dt = self.H(V2) * (1 - Si2) / self.taurse_i - Si2 / self.taudec_i

        return np.array([dV1dt, dn1dt, dSi1dt, dV2dt, dn2dt, dSi2dt])


    #integrador
    '''
    #Método de Runge-Kutta 4
    def rk4_step(self, y, t, dt):
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    '''
    def rk4_step(self, f, y, t, dt): #(En realidad es RK2 pero use mismo nombre para q sea mas facil)
        k1 = f(y, t)
        k2 = f(y + dt*k1, t + dt)
        return y + (dt/2)*(k1+k2)

##### Método que simula toda la red
##### Se accede con:  t, Y = red.simular_red()
    def simular_red(self, Tmax=2000, dt=0.01, Y0=None):
        t = np.arange(0, Tmax+dt, dt)
        Y = np.zeros((len(t), 6))  # Columnas: V1, n1, Si1, V2, n2, Si2
        if Y0 is None:
            Y[0, :] = [-65, 0, 0, -65, 0, 0]  
        else:
            Y[0, :] = Y0

        #Y Si quiero cambiar las cond inci: Y0 = [-70, 0.1, 0, -65, 0, 0]
        # t, Y = neurona1.simular_red(Y0=Y0)

        for i in range(1, len(t)):
            Y[i] = self.rk4_step(self.network_reduced_deriv, Y[i-1], t[i-1], dt)
            # Mantener neurona 2 en reposo hasta tappl2
            if t[i] < self.tappl2:
                Y[i, 3] = Y[0, 3]  # V2
                Y[i, 4] = Y[0, 4]  # n2
                Y[i, 5] = Y[0, 5]  # Si2
        return t, Y
    

##### CURVAS DE ACTIVACION/INACTIVACION Y CONSTANTES DE TIEMPO. 
#### Se accede con .graficar_curvas()
    def graficar_curvas(self):
        vv = np.arange(-100, 100.1, 0.1)
        
        fig, axs = plt.subplots(1, 2, figsize=(14,6))  # 1 fila, 2 columnas
        
        # Primera figura: curvas de activación/inactivación
        axs[0].plot(vv, self.minf(vv), 'b', linewidth=2, label=r'$m_{\infty}$')
        axs[0].plot(vv, self.hinf(vv), 'r', linewidth=2, label=r'$h_{\infty}$')
        axs[0].plot(vv, self.ninf(vv), 'g', linewidth=2, label=r'$n_{\infty}$')
        axs[0].set_xlim([-100, 120])
        axs[0].set_ylim([-0.1, 1.1])
        axs[0].set_xlabel('V', fontsize=14)
        axs[0].set_title('Curvas m,h,n', fontsize=16)
        axs[0].legend(fontsize=12)
        axs[0].grid(True)
        
        # Segunda figura: constantes de tiempo
        axs[1].plot(vv, self.taum(vv), 'b', linewidth=2, label=r'$\tau_m$')
        axs[1].plot(vv, self.tauh(vv), 'r', linewidth=2, label=r'$\tau_h$')
        axs[1].plot(vv, self.taun(vv), 'g', linewidth=2, label=r'$\tau_n$')
        axs[1].set_xlim([-100, 120])
        axs[1].set_ylim([-0.1, 12])
        axs[1].set_xlabel('V', fontsize=14)
        axs[1].set_title('Constantes de tiempo', fontsize=16)
        axs[1].legend(fontsize=12)
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()


##### Potenciales de membrana de las dos neuronas a lo largo del tiempo. 
##### Se accede con .graficar_potenciales_de_membrana_a_lo_largo_del_tiempo(t, Y)
    def graficar_potenciales_de_membrana_a_lo_largo_del_tiempo(self, t, Y):
        V1, V2 = Y[:, 0], Y[:, 3]
        plt.figure(figsize=(12, 4))
        plt.plot(t, V1, '-b', linewidth=2, label='V₁')
        plt.plot(t, V2, '-r', linewidth=2, label='V₂')
        plt.axvline(x=self.tconn, color=(0.7,0.7,0.7), linestyle='--', linewidth=2)
        plt.axis([0, t[-1], -80, 80])
        plt.xlabel('time [ms]', fontsize=14)
        plt.ylabel('Membrane Potential [mV]', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()


##### Comportamiento dinámico de cada neurona en el espacio de fases, útil para ver estabilidad y ciclos límite.
##### Se accede con .graficar_nullclines(Y, t[1]-t[0])
    def graficar_nullclines(self, Y, dt):
        jconn = int(self.tconn / dt)

        V1, n1 = Y[:, 0], Y[:, 1]
        V2, n2 = Y[:, 3], Y[:, 4]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 fila, 2 columnas

        # Nullcline V1-n1
        axes[0].plot(V1[jconn:], n1[jconn:], '-b', linewidth=2)
        axes[0].set_xlabel('V₁', fontsize=14)
        axes[0].set_ylabel('n₁', fontsize=14)
        axes[0].grid(True)
        axes[0].set_title('Nullcline V1-n1')

        # Nullcline V2-n2
        axes[1].plot(V2[jconn:], n2[jconn:], '-r', linewidth=2)
        axes[1].set_xlabel('V₂', fontsize=14)
        axes[1].set_ylabel('n₂', fontsize=14)
        axes[1].grid(True)
        axes[1].set_title('Nullcline V2-n2')

        plt.tight_layout()
        plt.show()