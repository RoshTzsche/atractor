import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.integrate import odeint

# 1. EL SISTEMA OCULTO (El "Cerebro" real de alta dimensión)
def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Simulamos 5000 milisegundos de dinámica
t = np.linspace(0, 50, 5000)
estado_inicial = [1.0, 1.0, 1.0]
sistema_real = odeint(lorenz_system, estado_inicial, t)

x_real = sistema_real[:, 0]
y_real = sistema_real[:, 1]
z_real = sistema_real[:, 2]

# 2. LA OBSERVACIÓN (El Electrodo colapsa todo a 1D)
s = x_real 

# 3. PARÁMETROS INICIALES DE TAKENS
tau_default = 12  
m = 3             

# 4. CONFIGURACIÓN VISUAL Y ESTRUCTURA (Negro puro y proporciones)
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('black') # Fondo de la ventana negro puro
# GridSpec: Asignamos más ancho a los atractores (1.5) y menos a la señal (0.7)
# bottom=0.25 deja espacio libre en la parte inferior para los controles
gs = gridspec.GridSpec(1, 3, width_ratios=[1.7, 0.4, 1.7], bottom=0.25)

# Plot 1: El Sistema Real Oculto
ax1 = fig.add_subplot(gs[0], projection='3d')
ax1.set_facecolor('black')
ax1.plot(x_real, y_real, z_real, color='#06d6a0', lw=0.5)
ax1.set_title("1. Sistema Real Oculto (3D)", color='#06d6a0', pad=10)
ax1.axis('off')

# Plot 2: Señal Observada (Reducida en tamaño horizontal)
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('black')
ax2.plot(t[:1500], s[:1500], color='#f72585', lw=1)
ax2.set_title("2. Señal Observada $s(t)$", color='#f72585')
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Voltaje")
ax2.grid(True, alpha=0.15) # Grid sutil para no opacar el fondo negro
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: Reconstrucción Geométrica (Atractor Reconstruido)
ax3 = fig.add_subplot(gs[2], projection='3d')
ax3.set_facecolor('black')
ax3.set_title(f"3. Reconstrucción Takens (τ={tau_default})", color='#00d4ff', pad=10)
ax3.axis('off')

# Calculamos el estado inicial de Takens y guardamos la referencia a la línea
X_t = s[:-2*tau_default]
Y_t = s[tau_default:-tau_default]
Z_t = s[2*tau_default:]
line_takens, = ax3.plot(X_t, Y_t, Z_t, color='#00d4ff', lw=0.5)

# ==========================================
# 5. CONTROLES INTERACTIVOS (Slider y Botón)
# ==========================================

# Definimos las áreas para el slider y el botón en las coordenadas de la figura
ax_slider = plt.axes([0.25, 0.1, 0.4, 0.03], facecolor='black')
ax_button = plt.axes([0.7, 0.095, 0.1, 0.04])

# Creación del Slider para ajustar tau
slider_tau = Slider(
    ax=ax_slider,
    label='Retraso Temporal ($\\tau$)',
    valmin=1,
    valmax=150,           # Rango máximo de tau a explorar
    valinit=tau_default,
    valstep=1,            # Aseguramos que tau sea siempre entero
    color='#00d4ff'
)

# Creación del botón de reinicio
btn_reset = Button(
    ax=ax_button, 
    label='Reset τ', 
    color='#1a1a1a', 
    hovercolor='#333333'
)
btn_reset.label.set_color('white')
btn_reset.label.set_weight('bold')

# Función que se ejecuta cada vez que mueves el slider
def update(val):
    tau = int(slider_tau.val)
    
    # Recalculamos los vectores de retraso con el nuevo tau
    X_new = s[:-2*tau]
    Y_new = s[tau:-tau]
    Z_new = s[2*tau:]
    
    # Actualizamos los datos 3D en tiempo real (evita redibujar todo el subplot)
    line_takens.set_data_3d(X_new, Y_new, Z_new)
    ax3.set_title(f"3. Reconstrucción Takens (τ={tau})", color='#00d4ff', pad=10)
    
    # Renderizamos los cambios
    fig.canvas.draw_idle()

# Función que se ejecuta al presionar el botón
def reset_slider(event):
    slider_tau.set_val(tau_default)

# Vinculamos los eventos a sus respectivas funciones
slider_tau.on_changed(update)
btn_reset.on_clicked(reset_slider)

plt.show()
