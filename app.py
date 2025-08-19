import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stt
import altair as alt

# ---------- Funciones auxiliares ----------
def simular_media_knownvar(mu, sigma, n, sims, alpha, rng):
    muestras = rng.normal(mu, sigma, (sims, n))
    medias = muestras.mean(axis=1)
    se = sigma / np.sqrt(n)
    z = stt.norm.ppf(1 - alpha/2)
    l = medias - z*se
    u = medias + z*se
    return muestras, medias, l, u

def simular_media_unknownvar(mu, sigma, n, sims, alpha, rng):
    muestras = rng.normal(mu, sigma, (sims, n))
    medias = muestras.mean(axis=1)
    s = muestras.std(axis=1, ddof=1)
    se = s / np.sqrt(n)
    tval = stt.t.ppf(1 - alpha/2, df=n-1)
    l = medias - tval*se
    u = medias + tval*se
    return muestras, medias, l, u

def simular_varianza(sigma, n, sims, alpha, rng):
    muestras = rng.normal(0, sigma, (sims, n))
    s2 = muestras.var(axis=1, ddof=1)
    chi2_low = stt.chi2.ppf(alpha/2, df=n-1)
    chi2_high = stt.chi2.ppf(1 - alpha/2, df=n-1)
    l = (n-1)*s2/chi2_high
    u = (n-1)*s2/chi2_low
    return muestras, s2, l, u

def simular_proporcion(p, n, sims, alpha, rng):
    muestras = rng.binomial(1, p, (sims, n))
    phat = muestras.mean(axis=1)
    se = np.sqrt(phat*(1-phat)/n)
    z = stt.norm.ppf(1 - alpha/2)
    l = phat - z*se
    u = phat + z*se
    return muestras, phat, l, u

def plot_intervalos(medidas, l, u, verdadero, titulo):
    df = pd.DataFrame({
        "sim": np.arange(len(medidas)),
        "media": medidas,
        "l": l,
        "u": u
    })
    df["cubre"] = (df["l"] <= verdadero) & (df["u"] >= verdadero)

    base = alt.Chart(df).encode(
        y=alt.Y("sim:O", title="Simulación", axis=None)
    )

    intervalos = base.mark_rule(size=2).encode(
        x="l:Q",
        x2="u:Q",
        color=alt.condition("datum.cubre", alt.value("steelblue"), alt.value("red"))
    )

    puntos = base.mark_point(filled=True, size=40).encode(
        x="media:Q",
        color=alt.condition("datum.cubre", alt.value("steelblue"), alt.value("red"))
    )

    linea_mu = alt.Chart(pd.DataFrame({"mu":[verdadero]})).mark_rule(strokeDash=[4,2], color="black").encode(
        x="mu:Q"
    )

    chart = (intervalos + puntos + linea_mu).properties(
        width=600, height=400, title=titulo
    )
    return chart, df

# ---------- App ----------
st.title("Simulador de Intervalos de Confianza")

tipo = st.sidebar.selectbox(
    "Elegí el tipo de intervalo",
    ["Media (σ conocida)", "Media (σ desconocida)", "Varianza", "Proporción"]
)

n = st.sidebar.number_input("Tamaño muestral (n)", 5, 500, 30, 1)
sims = st.sidebar.number_input("Número de simulaciones", 10, 500, 100, 10)
conf = st.sidebar.slider("Nivel de confianza", 0.80, 0.99, 0.95, 0.01)
alpha = 1 - conf

# clave para cachear datos
key = f"{tipo}-{n}-{sims}"

if "cache" not in st.session_state:
    st.session_state.cache = {}

rng = np.random.default_rng(1234)

# Generar o recuperar simulaciones
if key not in st.session_state.cache:
    if tipo == "Media (σ conocida)":
        mu = st.sidebar.number_input("Media poblacional (μ)", -100.0, 100.0, 0.0, 1.0)
        sigma = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medidas, l, u = simular_media_knownvar(mu, sigma, n, sims, alpha, rng)
        verdadero = mu
    elif tipo == "Media (σ desconocida)":
        mu = st.sidebar.number_input("Media poblacional (μ)", -100.0, 100.0, 0.0, 1.0)
        sigma = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medidas, l, u = simular_media_unknownvar(mu, sigma, n, sims, alpha, rng)
        verdadero = mu
    elif tipo == "Varianza":
        sigma = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0, 0.1)
        muestras, medidas, l, u = simular_varianza(sigma, n, sims, alpha, rng)
        verdadero = sigma**2
    else:  # Proporción
        p = st.sidebar.slider("Proporción poblacional (p)", 0.01, 0.99, 0.5, 0.01)
        muestras, medidas, l, u = simular_proporcion(p, n, sims, alpha, rng)
        verdadero = p

    st.session_state.cache[key] = (muestras, medidas, l, u, verdadero)
else:
    muestras, medidas, l, u, verdadero = st.session_state.cache[key]
    # Recalcular solo intervalos si cambia el nivel de confianza
    if tipo.startswith("Media (σ conocida)"):
        se = st.sidebar.number_input("Desvío poblacional (σ)", 0.1, 50.0, 5.0, 0.1) / np.sqrt(n)
        z = stt.norm.ppf(1 - alpha/2)
        l = medidas - z*se
        u = medidas + z*se
    elif tipo.startswith("Media (σ desconocida)"):
        s = muestras.std(axis=1, ddof=1)
        se = s / np.sqrt(n)
        tval = stt.t.ppf(1 - alpha/2, df=n-1)
        l = medidas - tval*se
        u = medidas + tval*se
    elif tipo == "Varianza":
        chi2_low = stt.chi2.ppf(alpha/2, df=n-1)
        chi2_high = stt.chi2.ppf(1 - alpha/2, df=n-1)
        l = (n-1)*medidas/chi2_high
        u = (n-1)*medidas/chi2_low
    else:  # Proporción
        se = np.sqrt(medidas*(1-medidas)/n)
        z = stt.norm.ppf(1 - alpha/2)
        l = medidas - z*se
        u = medidas + z*se

chart, df = plot_intervalos(medidas, l, u, verdadero, tipo)
st.altair_chart(chart, use_container_width=True)

st.subheader("Primeros 20 conjuntos muestrales")
df_muestras = pd.DataFrame(muestras[:20])
st.dataframe(df_muestras)
