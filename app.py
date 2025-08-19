import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm, t, chi2

# --- Funciones de simulación ---
def sim_media_sigma_conocida(mu, sigma, n, conf, sims, seed=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(mu, sigma, size=(sims, n))
    medias = data.mean(axis=1)
    z = norm.ppf(1 - (1-conf)/2)
    se = sigma / np.sqrt(n)
    lower = medias - z * se
    upper = medias + z * se
    return data, lower, upper

def sim_media_sigma_desconocida(mu, sigma, n, conf, sims, seed=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(mu, sigma, size=(sims, n))
    medias = data.mean(axis=1)
    s = data.std(axis=1, ddof=1)
    tval = t.ppf(1 - (1-conf)/2, df=n-1)
    se = s / np.sqrt(n)
    lower = medias - tval * se
    upper = medias + tval * se
    return data, lower, upper

def sim_varianza(sigma2, n, conf, sims, seed=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, np.sqrt(sigma2), size=(sims, n))
    s2 = data.var(axis=1, ddof=1)
    chi2_lower = chi2.ppf((1-conf)/2, df=n-1)
    chi2_upper = chi2.ppf(1-(1-conf)/2, df=n-1)
    lower = (n-1)*s2/chi2_upper
    upper = (n-1)*s2/chi2_lower
    return data, lower, upper

def sim_proporcion(p, n, conf, sims, seed=None):
    rng = np.random.default_rng(seed)
    data = rng.binomial(1, p, size=(sims, n))
    phat = data.mean(axis=1)
    z = norm.ppf(1 - (1-conf)/2)
    se = np.sqrt(phat*(1-phat)/n)
    lower = phat - z*se
    upper = phat + z*se
    return data, lower, upper

# --- Configuración de la app ---
st.title("Simulador de Intervalos de Confianza")

tipo = st.sidebar.selectbox(
    "Elegí el tipo de intervalo:",
    [
        "Media (σ conocida)",
        "Media (σ desconocida)",
        "Varianza",
        "Proporción"
    ]
)

# Parámetros comunes
conf = st.sidebar.slider("Nivel de confianza", 0.80, 0.99, 0.95, 0.01)
n = st.sidebar.slider("Tamaño muestral (n)", 5, 200, 30, 1)
sims = st.sidebar.slider("Número de simulaciones", 10, 500, 100, 10)

# --- Simulación según el tipo elegido ---
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.lower = None
    st.session_state.upper = None
    st.session_state.params = None

regen = False

if tipo == "Media (σ conocida)":
    mu = st.sidebar.number_input("Media poblacional (μ)", -100.0, 100.0, 0.0)
    sigma = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0)
    params = (tipo, mu, sigma, n, sims)
    if st.session_state.params != params:
        data, lower, upper = sim_media_sigma_conocida(mu, sigma, n, conf, sims)
        st.session_state.data, st.session_state.lower, st.session_state.upper = data, lower, upper
        st.session_state.params = params

elif tipo == "Media (σ desconocida)":
    mu = st.sidebar.number_input("Media poblacional (μ)", -100.0, 100.0, 0.0)
    sigma = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0)
    params = (tipo, mu, sigma, n, sims)
    if st.session_state.params != params:
        data, lower, upper = sim_media_sigma_desconocida(mu, sigma, n, conf, sims)
        st.session_state.data, st.session_state.lower, st.session_state.upper = data, lower, upper
        st.session_state.params = params

elif tipo == "Varianza":
    sigma2 = st.sidebar.number_input("Varianza poblacional (σ²)", 0.1, 50.0, 5.0)
    params = (tipo, sigma2, n, sims)
    if st.session_state.params != params:
        data, lower, upper = sim_varianza(sigma2, n, conf, sims)
        st.session_state.data, st.session_state.lower, st.session_state.upper = data, lower, upper
        st.session_state.params = params

elif tipo == "Proporción":
    p = st.sidebar.slider("Proporción poblacional (p)", 0.01, 0.99, 0.5, 0.01)
    params = (tipo, p, n, sims)
    if st.session_state.params != params:
        data, lower, upper = sim_proporcion(p, n, conf, sims)
        st.session_state.data, st.session_state.lower, st.session_state.upper = data, lower, upper
        st.session_state.params = params

# --- Recalcular CI si solo cambió el nivel de confianza ---
if tipo.startswith("Media"):
    if "sigma" in locals():
        _, lower, upper = sim_media_sigma_conocida(mu, sigma, n, conf, sims) if tipo=="Media (σ conocida)" else sim_media_sigma_desconocida(mu, sigma, n, conf, sims)
        st.session_state.lower, st.session_state.upper = lower, upper
elif tipo == "Varianza":
    _, lower, upper = sim_varianza(sigma2, n, conf, sims)
    st.session_state.lower, st.session_state.upper = lower, upper
elif tipo == "Proporción":
    _, lower, upper = sim_proporcion(p, n, conf, sims)
    st.session_state.lower, st.session_state.upper = lower, upper

# --- Visualización ---
df = pd.DataFrame({
    "sim": np.arange(sims),
    "lower": st.session_state.lower,
    "upper": st.session_state.upper
})

true_value = None
if tipo.startswith("Media"):
    true_value = mu
elif tipo == "Varianza":
    true_value = sigma2
elif tipo == "Proporción":
    true_value = p

x_min = df["lower"].min()
x_max = df["upper"].max()
margin = (x_max - x_min) * 0.1
x_min -= margin
x_max += margin

chart = alt.Chart(df).mark_rule().encode(
    x="lower:Q",
    x2="upper:Q",
    y=alt.Y("sim:O", sort="descending"),
    color=alt.condition(
        (alt.datum.lower <= true_value) & (alt.datum.upper >= true_value),
        alt.value("steelblue"),
        alt.value("red")
    )
).properties(width=600, height=400)

line = alt.Chart(pd.DataFrame({"x":[true_value]})).mark_rule(color="green").encode(x="x")

st.altair_chart(chart + line, use_container_width=True)

# --- Datos de muestra ---
st.subheader("Primeros 20 conjuntos muestrales")
sample_df = pd.DataFrame(st.session_state.data[:20])
st.dataframe(sample_df)
