import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy import stats

# =========================
# Funciones auxiliares
# =========================

def generar_muestras(n, sims, dist, media, sigma=None, p=None):
    """Genera los datos muestrales según el tipo de distribución."""
    data = []
    for i in range(sims):
        if dist == "Normal (varianza conocida)" or dist == "Normal (varianza desconocida)":
            muestra = np.random.normal(loc=media, scale=sigma, size=n)
            data.append(muestra)
        elif dist == "Binomial (proporción)":
            muestra = np.random.binomial(1, p, size=n)
            data.append(muestra)
    return data

def calcular_intervalos(data, dist, media, sigma, p, alpha):
    """Calcula intervalos de confianza según el tipo elegido."""
    resultados = []
    for i, muestra in enumerate(data, start=1):  # ahora empieza en 1
        n = len(muestra)
        xbar = np.mean(muestra)

        if dist == "Normal (varianza conocida)":
            se = sigma / np.sqrt(n)
            z = stats.norm.ppf(1 - alpha/2)
            li, ls = xbar - z*se, xbar + z*se
            contiene = li <= media <= ls
            resultados.append([i, xbar, li, ls, contiene])

        elif dist == "Normal (varianza desconocida)":
            s = np.std(muestra, ddof=1)
            se = s / np.sqrt(n)
            t = stats.t.ppf(1 - alpha/2, df=n-1)
            li, ls = xbar - t*se, xbar + t*se
            contiene = li <= media <= ls
            resultados.append([i, xbar, li, ls, contiene])

        elif dist == "Binomial (proporción)":
            phat = np.mean(muestra)
            se = np.sqrt(phat*(1-phat)/n)
            z = stats.norm.ppf(1 - alpha/2)
            li, ls = phat - z*se, phat + z*se
            contiene = li <= p <= ls
            resultados.append([i, phat, li, ls, contiene])

    return pd.DataFrame(resultados, columns=["Simulación", "Estadístico", "LI", "LS", "Contiene"])

# =========================
# Interfaz Streamlit
# =========================

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")
st.title("🔎 Simulador de Intervalos de Confianza")

# Menú lateral
st.sidebar.header("Parámetros de la simulación")
tipo = st.sidebar.selectbox(
    "Elegí el tipo de intervalo:",
    ["Normal (varianza conocida)", "Normal (varianza desconocida)", "Binomial (proporción)"]
)

n = st.sidebar.slider("Tamaño muestral (n)", min_value=2, max_value=500, value=30, step=1)
sims = st.sidebar.slider("Cantidad de simulaciones", min_value=1, max_value=200, value=50, step=1)
conf = st.sidebar.slider("Nivel de confianza (%)", min_value=80, max_value=99, value=95, step=1)
alpha = 1 - conf/100

if "data" not in st.session_state or st.session_state.n != n or st.session_state.sims != sims or st.session_state.tipo != tipo:
    # Generar nuevas simulaciones solo si cambian n, sims o tipo de intervalo
    if tipo.startswith("Normal"):
        media = st.sidebar.number_input("Media poblacional", value=0.0)
        sigma = st.sidebar.number_input("Desvío estándar poblacional (σ)", value=1.0, min_value=0.0001)
        data = generar_muestras(n, sims, tipo, media, sigma=sigma)
        st.session_state.media, st.session_state.sigma = media, sigma
        st.session_state.p = None
    else:
        p = st.sidebar.slider("Proporción poblacional (p)", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
        data = generar_muestras(n, sims, tipo, media=None, p=p)
        st.session_state.p = p
        st.session_state.media, st.session_state.sigma = None, None

    st.session_state.data = data
    st.session_state.n = n
    st.session_state.sims = sims
    st.session_state.tipo = tipo
else:
    data = st.session_state.data
    media, sigma, p = st.session_state.media, st.session_state.sigma, st.session_state.p

# Calcular intervalos
df = calcular_intervalos(data, tipo, media, sigma, p, alpha)

# =========================
# Gráfico Altair
# =========================
c = alt.Chart(df).mark_rule().encode(
    x="LI",
    x2="LS",
    y=alt.Y("Simulación:O", sort="descending"),
    color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
).properties(
    width=700,  # ancho fijo -> sin desplazamiento horizontal
    height=400
)

point = alt.Chart(df).mark_point(filled=True, size=30).encode(
    x="Estadístico",
    y="Simulación:O",
    color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
)

# Línea vertical en el parámetro poblacional (fijo en pantalla)
if tipo.startswith("Normal"):
    linea = alt.Chart(pd.DataFrame({"valor": [media]})).mark_rule(color="green", strokeDash=[4,4]).encode(x="valor")
elif tipo == "Binomial (proporción)":
    linea = alt.Chart(pd.DataFrame({"valor": [p]})).mark_rule(color="green", strokeDash=[4,4]).encode(x="valor")

st.altair_chart(c + point + linea, use_container_width=True)

# =========================
# Tabla de los primeros 20 conjuntos muestrales
# =========================
st.subheader("Primeros 20 conjuntos muestrales")
primeros = pd.DataFrame(data[:20]).T
st.dataframe(primeros)
