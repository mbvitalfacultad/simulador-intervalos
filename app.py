# -*- coding: utf-8 -*-
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
    li = medias - z*se
    ls = medias + z*se
    contiene = (li <= mu) & (ls >= mu)
    return muestras, medias, li, ls, contiene

def sim_media_sigma_desconocida(mu, sigma, n, sims, alpha, rng):
    muestras = rng.normal(mu, sigma, (sims, n))
    medias = muestras.mean(axis=1)
    s = muestras.std(axis=1, ddof=1)
    se = s / np.sqrt(n)
    tval = t.ppf(1 - alpha/2, df=n-1)
    li = medias - tval*se
    ls = medias + tval*se
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
    # evitar sqrt de negativo por cuestiones numéricas
    se = np.sqrt(np.maximum(phat*(1-phat)/n, 0.0))
    z = norm.ppf(1 - alpha/2)
    li = phat - z*se
    ls = phat + z*se
    contiene = (li <= p) & (ls >= p)
    return muestras, phat, li, ls, contiene

# =========================
# App
# =========================

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")
st.title("Simulador de Intervalos de Confianza")  # eliminado emoji para evitar posibles errores de encoding

# Sidebar - control siempre visible
st.sidebar.header("Parámetros de la simulación")
tipo = st.sidebar.selectbox(
    "Tipo de intervalo",
    ["Media con varianza conocida", "Media con varianza desconocida", "Varianza", "Proporción"]
)

n = st.sidebar.slider("Tamaño muestral (n)", 2, 500, 30, 1)
sims = st.sidebar.slider("Número de simulaciones", 1, 200, 50, 1)
conf = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95, 1)
alpha = 1 - conf/100

# Persistencia de parámetros en session_state (valores iniciales)
if "mu" not in st.session_state:
    st.session_state.mu = 0.0
if "sigma" not in st.session_state:
    st.session_state.sigma = 5.0
if "p" not in st.session_state:
    st.session_state.p = 0.5

# Sliders de parámetros poblacionales siempre visibles y con valores almacenados
if tipo in ["Media con varianza conocida", "Media con varianza desconocida"]:
    mu = st.sidebar.slider("Media poblacional (μ)", -100.0, 100.0, value=st.session_state.mu, step=0.1, key="mu_slider")
    sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1, key="sigma_slider")
    st.session_state.mu = mu
    st.session_state.sigma = sigma
elif tipo == "Varianza":
    sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1, key="sigma_slider_var")
    st.session_state.sigma = sigma
elif tipo == "Proporción":
    p = st.sidebar.slider("Proporción poblacional (p)", 0.01, 0.99, value=st.session_state.p, step=0.01, key="p_slider")
    st.session_state.p = p

# RNG (fijo para reproducibilidad)
rng = np.random.default_rng(1234)

# Ejecutar la simulación según el tipo (estadístico devuelto se nombra 'estadistico')
if tipo == "Media con varianza conocida":
    muestras, estadistico, li, ls, contiene = sim_media_sigma_conocida(st.session_state.mu, st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.mu
elif tipo == "Media con varianza desconocida":
    muestras, estadistico, li, ls, contiene = sim_media_sigma_desconocida(st.session_state.mu, st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.mu
elif tipo == "Varianza":
    muestras, estadistico, li, ls, contiene = sim_varianza(st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.sigma**2
elif tipo == "Proporción":
    muestras, estadistico, li, ls, contiene = sim_proporcion(st.session_state.p, n, sims, alpha, rng)
    valor_real = st.session_state.p

# DataFrame para Altair (cada fila = una simulación)
df = pd.DataFrame({
    "Simulación": np.arange(1, sims+1),
    "Estadístico": estadistico,
    "LI": li,
    "LS": ls,
    "Contiene": contiene
})

# Dominio fijo (mínimo/máximo entre LI, LS y el valor poblacional) + pequeño padding
xmin = min(df["LI"].min(), df["Estadístico"].min(), valor_real)
xmax = max(df["LS"].max(), df["Estadístico"].max(), valor_real)
span = xmax - xmin
if span == 0:
    pad = 1.0
else:
    pad = span * 0.05
xmin_pad = xmin - pad
xmax_pad = xmax + pad

# Función de trazado con dominio fijo y línea verde de referencia
def plot_intervalos(df, valor_parametro, titulo, xmin_pad, xmax_pad):
    base = alt.Chart(df).encode(
        y=alt.Y("Simulación:O", sort="descending", title=None)
    )

    intervalos = base.mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=[xmin_pad, xmax_pad]), title=None),
        x2="LS:Q",
        color=alt.condition(alt.datum.Contiene, alt.value("steelblue"), alt.value("red")),
        tooltip=[
            alt.Tooltip("Simulación:Q"),
            alt.Tooltip("LI:Q", title="LI"),
            alt.Tooltip("LS:Q", title="LS"),
            alt.Tooltip("Contiene:Q", title="Contiene")
        ]
    )

    puntos = base.mark_point(filled=True, size=60).encode(
        x=alt.X("Estadístico:Q", scale=alt.Scale(domain=[xmin_pad, xmax_pad]), title=None),
        color=alt.condition(alt.datum.Contiene, alt.value("steelblue"), alt.value("red")),
        tooltip=[
            alt.Tooltip("Simulación:Q"),
            alt.Tooltip("Estadístico:Q", title="Estadístico"),
            alt.Tooltip("Contiene:Q", title="Contiene")
        ]
    )

    linea = alt.Chart(pd.DataFrame({"valor":[valor_parametro]})).mark_rule(color="green", strokeDash=[4,4], size=2).encode(
        x=alt.X("valor:Q", scale=alt.Scale(domain=[xmin_pad, xmax_pad]), title=None)
    )

    # usar layer para que compartan el mismo dominio x
    chart = alt.layer(intervalos, puntos, linea).properties(
        height=600,
        title=alt.TitleParams(text=titulo, anchor="start", fontSize=16)
    ).configure_title(
        anchor="start"
    )
    return chart

# Mostrar gráfico
chart = plot_intervalos(df, valor_real, tipo, xmin_pad, xmax_pad)
st.altair_chart(chart, use_container_width=True)

# Mostrar primeros 20 conjuntos muestrales (index desde 1)
primeras_20 = pd.DataFrame(muestras[:20])
primeras_20.index = np.arange(1, primeras_20.shape[0]+1)
primeras_20.columns = np.arange(1, primeras_20.shape[1]+1)
st.subheader("Primeros 20 conjuntos muestrales")
st.dataframe(primeras_20)
