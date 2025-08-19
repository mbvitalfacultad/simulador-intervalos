import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# --- TÍTULO ---
st.title("Simulador de Intervalos de Confianza")

# --- SELECCIÓN DE TIPO DE INTERVALO ---
tipo_intervalo = st.selectbox(
    "Selecciona el tipo de intervalo de confianza:",
    [
        "Media (σ conocida)",
        "Media (σ desconocida)",
        "Varianza",
        "Proporción"
    ]
)

# --- PARÁMETROS POBLACIONALES (persisten en session_state) ---
if "params" not in st.session_state:
    st.session_state.params = {
        "media": 10.0,
        "sigma": 5.0,
        "varianza": 25.0,
        "p": 0.5
    }

# --- Función para número + deslizador ---
def number_slider(label, key, min_val, max_val, step, default):
    c1, c2 = st.columns([1, 3])
    with c1:
        num = st.number_input(label, value=st.session_state.params[key], step=step, key=f"{key}_num")
    with c2:
        slider = st.slider(" ", min_val, max_val, value=float(st.session_state.params[key]), step=step, key=f"{key}_slider")
    val = num if num != st.session_state.params[key] else slider
    st.session_state.params[key] = val
    return val

# --- Selección de parámetros según el tipo ---
if "Media" in tipo_intervalo:
    media = number_slider("Media poblacional", "media", -50.0, 50.0, 0.1, 10.0)
    sigma = number_slider("Desvío estándar poblacional", "sigma", 0.1, 20.0, 0.1, 5.0)
elif tipo_intervalo == "Varianza":
    varianza = number_slider("Varianza poblacional", "varianza", 0.1, 100.0, 0.1, 25.0)
elif tipo_intervalo == "Proporción":
    p = number_slider("Proporción poblacional", "p", 0.01, 0.99, 0.01, 0.5)

# --- Parámetros comunes ---
n = st.number_input("Tamaño de muestra (n)", min_value=2, max_value=500, value=30, step=1)
confianza = st.slider("Nivel de confianza (%)", 80, 99, 95, step=1)
simulaciones = st.number_input("Número de simulaciones", min_value=1, max_value=200, value=20, step=1)

alpha = 1 - confianza / 100

# --- FUNCIÓN DE SIMULACIÓN ---
def generar_intervalos():
    intervalos = []
    contiene = []
    valor_real = None

    if tipo_intervalo == "Media (σ conocida)":
        valor_real = st.session_state.params["media"]
        for _ in range(simulaciones):
            muestra = np.random.normal(valor_real, st.session_state.params["sigma"], n)
            xbar = np.mean(muestra)
            z = stats.norm.ppf(1 - alpha / 2)
            li = xbar - z * st.session_state.params["sigma"] / np.sqrt(n)
            ls = xbar + z * st.session_state.params["sigma"] / np.sqrt(n)
            intervalos.append((li, ls))
            contiene.append(li <= valor_real <= ls)

    elif tipo_intervalo == "Media (σ desconocida)":
        valor_real = st.session_state.params["media"]
        for _ in range(simulaciones):
            muestra = np.random.normal(valor_real, st.session_state.params["sigma"], n)
            xbar = np.mean(muestra)
            s = np.std(muestra, ddof=1)
            t = stats.t.ppf(1 - alpha / 2, df=n - 1)
            li = xbar - t * s / np.sqrt(n)
            ls = xbar + t * s / np.sqrt(n)
            intervalos.append((li, ls))
            contiene.append(li <= valor_real <= ls)

    elif tipo_intervalo == "Varianza":
        valor_real = st.session_state.params["varianza"]
        for _ in range(simulaciones):
            muestra = np.random.normal(0, np.sqrt(valor_real), n)
            s2 = np.var(muestra, ddof=1)
            chi2_inf = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
            chi2_sup = stats.chi2.ppf(alpha / 2, df=n - 1)
            li = (n - 1) * s2 / chi2_inf
            ls = (n - 1) * s2 / chi2_sup
            intervalos.append((li, ls))
            contiene.append(li <= valor_real <= ls)

    elif tipo_intervalo == "Proporción":
        valor_real = st.session_state.params["p"]
        for _ in range(simulaciones):
            muestra = np.random.binomial(1, valor_real, n)
            phat = np.mean(muestra)
            z = stats.norm.ppf(1 - alpha / 2)
            se = np.sqrt(phat * (1 - phat) / n)
            li = phat - z * se
            ls = phat + z * se
            li, ls = max(0, li), min(1, ls)
            intervalos.append((li, ls))
            contiene.append(li <= valor_real <= ls)

    return intervalos, contiene, valor_real

# --- GENERACIÓN Y GRÁFICO ---
intervalos, contiene, valor_real = generar_intervalos()

fig, ax = plt.subplots(figsize=(8, 6))

for i, (interv, ok) in enumerate(zip(intervalos, contiene)):
    li, ls = interv
    ax.plot([li, ls], [i, i], color="blue" if ok else "red", lw=2)

# Línea verde fija en el valor poblacional
ax.axvline(x=valor_real, color="green", linestyle="--", lw=2)

ax.set_title(f"Intervalos de confianza para {tipo_intervalo}", fontsize=14)
ax.set_xlabel("Valor del parámetro")
ax.set_yticks([])

st.pyplot(fig)
