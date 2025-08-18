import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm, t, chi2

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="centered")

st.title("📊 Simulador de Intervalos de Confianza")

st.sidebar.header("Parámetros")

tipo_ic = st.sidebar.selectbox(
    "Selecciona el tipo de intervalo",
    ["Media (σ conocida)", "Media (σ desconocida)", "Varianza", "Proporción"]
)

n = st.sidebar.slider("Tamaño muestral (n)", 5, 500, 30)
confianza = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95)
alpha = 1 - confianza/100

# Datos simulados
np.random.seed(42)
datos = np.random.normal(loc=50, scale=10, size=n)
media_muestral = np.mean(datos)
s = np.std(datos, ddof=1)
p_hat = np.mean(np.random.binomial(1, 0.5, n))

if tipo_ic == "Media (σ conocida)":
    sigma = 10
    z = norm.ppf(1 - alpha/2)
    li = media_muestral - z * sigma / np.sqrt(n)
    ls = media_muestral + z * sigma / np.sqrt(n)
    st.write(f"IC para la media (σ conocida): ({li:.2f}, {ls:.2f})")

elif tipo_ic == "Media (σ desconocida)":
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    li = media_muestral - t_crit * s / np.sqrt(n)
    ls = media_muestral + t_crit * s / np.sqrt(n)
    st.write(f"IC para la media (σ desconocida): ({li:.2f}, {ls:.2f})")

elif tipo_ic == "Varianza":
    chi2_inf = chi2.ppf(alpha/2, df=n-1)
    chi2_sup = chi2.ppf(1 - alpha/2, df=n-1)
    li = (n-1) * s**2 / chi2_sup
    ls = (n-1) * s**2 / chi2_inf
    st.write(f"IC para la varianza: ({li:.2f}, {ls:.2f})")

elif tipo_ic == "Proporción":
    z = norm.ppf(1 - alpha/2)
    li = p_hat - z * np.sqrt(p_hat*(1-p_hat)/n)
    ls = p_hat + z * np.sqrt(p_hat*(1-p_hat)/n)
    st.write(f"IC para la proporción: ({li:.2f}, {ls:.2f})")

# Visualización con Altair
df = pd.DataFrame({
    "Estimador": [media_muestral if "Media" in tipo_ic else p_hat],
    "LI": [li],
    "LS": [ls]
})

chart = alt.Chart(df).mark_errorbar(extent="ci").encode(
    x=alt.X("LI", title="Intervalo de Confianza"),
    x2="LS",
    y=alt.value(0)
) + alt.Chart(df).mark_point(size=100, color="red").encode(
    x="Estimador",
    y=alt.value(0)
)

st.altair_chart(chart, use_container_width=True)
