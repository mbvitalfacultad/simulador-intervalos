import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm, t, chi2

# =========================
# Funciones auxiliares
# =========================

def sim_media_sigma_conocida(mu, sigma, n, sims, alpha, rng):
    muestras = rng.normal(mu, sigma, (sims, n))
    medias = muestras.mean(axis=1)
    se = sigma / np.sqrt(n)
    z = norm.ppf(1 - alpha/2)
    li, ls = medias - z*se, medias + z*se
    contiene = (li <= mu) & (ls >= mu)
    return muestras, medias, li, ls, contiene

def sim_media_sigma_desconocida(mu, sigma, n, sims, alpha, rng):
    muestras = rng.normal(mu, sigma, (sims, n))
    medias = muestras.mean(axis=1)
    s = muestras.std(axis=1, ddof=1)
    se = s / np.sqrt(n)
    tval = t.ppf(1 - alpha/2, df=n-1)
    li, ls = medias - tval*se, medias + tval*se
    contiene = (li <= mu) & (ls >= mu)
    return muestras, medias, li, ls, contiene

def sim_varianza(sigma, n, sims, alpha, rng):
    muestras = rng.normal(0, sigma, (sims, n))
    s2 = muestras.var(axis=1, ddof=1)
    chi2_low = chi2.ppf(alpha/2, df=n-1)
    chi2_high = chi2.ppf(1 - alpha/2, df=n-1)
    li = (n-1)*s2/chi2_high
    ls = (n-1)*s2/chi2_low
    contiene = (li <= sigma**2) & (ls >= sigma**2)
    return muestras, s2, li, ls, contiene

def sim_proporcion(p, n, sims, alpha, rng):
    muestras = rng.binomial(1, p, (sims, n))
    phat = muestras.mean(axis=1)
    se = np.sqrt(phat*(1-phat)/n)
    z = norm.ppf(1 - alpha/2)
    li, ls = phat - z*se, phat + z*se
    contiene = (li <= p) & (ls >= p)
    return muestras, phat, li, ls, contiene

def plot_intervalos(df, valor_parametro, titulo):
    base = alt.Chart(df).encode(
        y=alt.Y("Simulación:O", sort="descending")
    )

    intervalos = base.mark_rule(size=2).encode(
        x="LI",
        x2="LS",
        color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
    )

    puntos = base.mark_point(filled=True, size=30).encode(
        x="Estadístico",
        color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
    )

    linea = alt.Chart(pd.DataFrame({"valor":[valor_parametro]})).mark_rule(color="green", strokeDash=[4,4]).encode(
        x="valor"
    )

    chart = (intervalos + puntos + linea).properties(width=700, height=400, title=titulo)
    return chart

# =========================
# App
# =========================

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")
st.title("🔎 Simulador de Intervalos de Confianza")

# Sidebar
st.sidebar.header("Parámetros de la simulación")
tipo = st.sidebar.selectbox(
    "Tipo de intervalo",
    ["Media con varianza conocida", "Media con varianza desconocida", "Varianza", "Proporción"]
)

n = st.sidebar.slider("Tamaño muestral (n)", 2, 500, 30, 1)
sims = st.sidebar.slider("Número de simulaciones", 1, 200, 50, 1)
conf = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95, 1)
alpha = 1 - conf/100

# Persistencia de datos
if "cache" not in st.session_state:
    st.session_state.cache = {}

key = f"{tipo}-{n}-{sims}"

rng = np.random.default_rng(1234)

if key not in st.session_state.cache:
    # Generar datos y guardar parámetros
    if tipo == "Media con varianza conocida":
        mu = st.sidebar.slider("Media poblacional (μ)", -100.0, 100.0, 0.0, 0.1)
        sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medias, li, ls, contiene = sim_media_sigma_conocida(mu, sigma, n, sims, alpha, rng)
        valor_real = mu
        st.session_state.cache[key] = (muestras, medias, li, ls, contiene, valor_real, mu, sigma, None)
        st.session_state.mu = mu
        st.session_state.sigma = sigma
    elif tipo == "Media con varianza desconocida":
        mu = st.sidebar.slider("Media poblacional (μ)", -100.0, 100.0, 0.0, 0.1)
        sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medias, li, ls, contiene = sim_media_sigma_desconocida(mu, sigma, n, sims, alpha, rng)
        valor_real = mu
        st.session_state.cache[key] = (muestras, medias, li, ls, contiene, valor_real, mu, sigma, None)
        st.session_state.mu = mu
        st.session_state.sigma = sigma
    elif tipo == "Varianza":
        sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medias, li, ls, contiene = sim_varianza(sigma, n, sims, alpha, rng)
        valor_real = sigma**2
        st.session_state.cache[key] = (muestras, medias, li, ls, contiene, valor_real, None, sigma, None)
        st.session_state.sigma = sigma
    elif tipo == "Proporción":
        p = st.sidebar.slider("Proporción poblacional (p)", 0.01, 0.99, 0.5, 0.01)
        muestras, medias, li, ls, contiene = sim_proporcion(p, n, sims, alpha, rng)
        valor_real = p
        st.session_state.cache[key] = (muestras, medias, li, ls, contiene, valor_real, None, None, p)
        st.session_state.p = p
else:
    muestras, medias, li, ls, contiene, valor_real, mu, sigma, p = st.session_state.cache[key]
    # Recalcular IC si cambia conf
    if tipo == "Media con varianza conocida":
        se = st.session_state.sigma / np.sqrt(n)
        z = norm.ppf(1-alpha/2)
        li = medias - z*se
        ls = medias + z*se
        contiene = (li <= st.session_state.mu) & (ls >= st.session_state.mu)
    elif tipo == "Media con varianza desconocida":
        s = muestras.std(axis=1, ddof=1)
        se = s / np.sqrt(n)
        tval = t.ppf(1-alpha/2, df=n-1)
        li = medias - tval*se
        ls = medias + tval*se
        contiene = (li <= st.session_state.mu) & (ls >= st.session_state.mu)
    elif tipo == "Varianza":
        s2 = muestras.var(axis=1, ddof=1)
        chi2_low = chi2.ppf(alpha/2, df=n-1)
        chi2_high = chi2.ppf(1 - alpha/2, df=n-1)
        li = (n-1)*s2/chi2_high
        ls = (n-1)*s2/chi2_low
        contiene = (li <= st.session_state.sigma**2) & (ls >= st.session_state.sigma**2)
    elif tipo == "Proporción":
        phat = muestras.mean(axis=1)
        se = np.sqrt(phat*(1-phat)/n)
        z = norm.ppf(1-alpha/2)
        li = phat - z*se
        ls = phat + z*se
        contiene = (li <= st.session_state.p) & (ls >= st.session_state.p)

# DataFrame para Altair
df = pd.DataFrame({
    "Simulación": np.arange(1, sims+1),
    "Estadístico": medias,
    "LI": li,
    "LS": ls,
    "Contiene": contiene
})

# Determinar valor poblacional para la línea
if tipo in ["Media con varianza conocida", "Media con varianza desconocida"]:
    valor_parametro = st.session_state.mu
elif tipo == "Varianza":
    valor_parametro = st.session_state.sigma**2
elif tipo == "Proporción":
    valor_parametro = st.session_state.p

# Gráfico
chart = plot_intervalos(df, valor_parametro, tipo)
st.altair_chart(chart, use_container_width=True)

# Tabla de primeros 20 conjuntos muestrales con numeración desde 1
primeras_20 = pd.DataFrame(muestras[:20])
primeras_20.index = np.arange(1, primeras_20.shape[0]+1)  # filas numeradas desde 1
primeras_20.columns = np.arange(1, primeras_20.shape[1]+1)  # columnas numeradas desde 1
st.subheader("Primeros 20 conjuntos muestrales")
st.dataframe(primeras_20)
