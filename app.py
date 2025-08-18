
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm, t, chi2

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")

st.title("🎯 Simulador de Intervalos de Confianza")

# --- Sidebar ---
st.sidebar.header("Parámetros de la simulación")
tipo = st.sidebar.selectbox("Elige el tipo de intervalo:", [
    "Media (varianza conocida)",
    "Media (varianza desconocida)",
    "Varianza",
    "Proporción"
])

n = st.sidebar.slider("Tamaño muestral (n)", 5, 200, 30)
conf = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95)
simulaciones = st.sidebar.slider("Número de simulaciones", 10, 500, 100)

# Parámetros poblacionales
if tipo == "Proporción":
    p0 = st.sidebar.slider("Proporción poblacional (p)", 0.05, 0.95, 0.5, 0.05)
else:
    mu = st.sidebar.number_input("Media poblacional (μ)", value=0.0)
    sigma = st.sidebar.number_input("Desvío estándar poblacional (σ)", value=1.0)

alpha = 1 - conf/100

# --- Simulación ---
resultados = []
for s in range(simulaciones):
    if tipo == "Media (varianza conocida)":
        datos = np.random.normal(mu, sigma, n)
        media_m = np.mean(datos)
        z = norm.ppf(1 - alpha/2)
        error = z * sigma / np.sqrt(n)
        li, ls = media_m - error, media_m + error
        cubre = (mu >= li) and (mu <= ls)
        estimador = media_m

    elif tipo == "Media (varianza desconocida)":
        datos = np.random.normal(mu, sigma, n)
        media_m = np.mean(datos)
        s_m = np.std(datos, ddof=1)
        tval = t.ppf(1 - alpha/2, df=n-1)
        error = tval * s_m / np.sqrt(n)
        li, ls = media_m - error, media_m + error
        cubre = (mu >= li) and (mu <= ls)
        estimador = media_m

    elif tipo == "Varianza":
        datos = np.random.normal(mu, sigma, n)
        s2 = np.var(datos, ddof=1)
        li = (n-1)*s2/chi2.ppf(1-alpha/2, df=n-1)
        ls = (n-1)*s2/chi2.ppf(alpha/2, df=n-1)
        cubre = (sigma**2 >= li) and (sigma**2 <= ls)
        estimador = s2

    elif tipo == "Proporción":
        datos = np.random.binomial(1, p0, n)
        phat = np.mean(datos)
        z = norm.ppf(1 - alpha/2)
        error = z * np.sqrt(phat*(1-phat)/n)
        li, ls = phat - error, phat + error
        cubre = (p0 >= li) and (p0 <= ls)
        estimador = phat

    resultados.append({
        "Simulación": s+1,
        "Datos muestrales": datos,
        "Estimador": estimador,
        "LI": li,
        "LS": ls,
        "Cubre": cubre
    })

df = pd.DataFrame(resultados)

# --- Gráfico ---
interval_chart = alt.Chart(df).mark_rule(size=2).encode(
    x="LI:Q",
    x2="LS:Q",
    y=alt.Y("Simulación:O", sort='descending'),
    color=alt.condition("datum.Cubre", alt.value("green"), alt.value("red"))
).properties(width=600, height=400, title="Intervalos de confianza simulados")

point_chart = alt.Chart(df).mark_point(filled=True, size=30, color="black").encode(
    x="Estimador:Q",
    y=alt.Y("Simulación:O", sort='descending')
)

st.altair_chart(interval_chart + point_chart, use_container_width=True)

# --- Tabla con los primeros 20 ---
st.subheader("📊 Primeros 20 conjuntos muestrales")
st.write(df.head(20)[["Simulación", "Datos muestrales", "Estimador", "LI", "LS", "Cubre"]])

# --- Descargar CSV completo ---
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar todos los resultados en CSV",
    data=csv,
    file_name="resultados_simulacion.csv",
    mime="text/csv"
)
