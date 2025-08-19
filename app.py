import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2

st.set_page_config(layout="wide")

# --- Función para simular intervalos ---
def simular_intervalos(tipo, mu, sigma, n, nivel_confianza, num_simulaciones, p=None):
    resultados = []
    alpha = 1 - nivel_confianza

    for _ in range(num_simulaciones):
        if tipo in ["media_varianza_conocida", "media_varianza_desconocida"]:
            muestra = np.random.normal(mu, sigma, n)
            media_muestral = np.mean(muestra)

            if tipo == "media_varianza_conocida":
                z = norm.ppf(1 - alpha/2)
                error = z * sigma / np.sqrt(n)
            else:
                s = np.std(muestra, ddof=1)
                t_val = t.ppf(1 - alpha/2, df=n-1)
                error = t_val * s / np.sqrt(n)

            li = media_muestral - error
            ls = media_muestral + error
            contiene = (li <= mu <= ls)
            resultados.append((li, ls, media_muestral, contiene))

        elif tipo == "varianza":
            muestra = np.random.normal(mu, sigma, n)
            s2 = np.var(muestra, ddof=1)
            chi2_inf = chi2.ppf(1 - alpha/2, df=n-1)
            chi2_sup = chi2.ppf(alpha/2, df=n-1)
            li = (n-1)*s2/chi2_inf
            ls = (n-1)*s2/chi2_sup
            contiene = (li <= sigma**2 <= ls)
            resultados.append((li, ls, s2, contiene))

        elif tipo == "proporcion":
            muestra = np.random.binomial(1, p, n)
            phat = np.mean(muestra)
            z = norm.ppf(1 - alpha/2)
            error = z * np.sqrt(phat*(1-phat)/n)
            li = phat - error
            ls = phat + error
            contiene = (li <= p <= ls)
            resultados.append((li, ls, phat, contiene))

    return resultados

# --- Controles de usuario ---
st.sidebar.header("Parámetros")

tipo = st.sidebar.selectbox(
    "Selecciona el tipo de intervalo",
    ["media_varianza_conocida", "media_varianza_desconocida", "varianza", "proporcion"]
)

nivel_confianza = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95) / 100
n = st.sidebar.number_input("Tamaño de muestra (n)", min_value=2, max_value=1000, value=30, step=1)
num_simulaciones = st.sidebar.number_input("Número de simulaciones", min_value=1, max_value=200, value=20, step=1)

if tipo in ["media_varianza_conocida", "media_varianza_desconocida"]:
    mu = st.sidebar.number_input("Media poblacional", value=0.0, step=0.1)
    sigma = st.sidebar.number_input("Desvío estándar poblacional", min_value=0.1, value=1.0, step=0.1)
    p = None
elif tipo == "varianza":
    mu = st.sidebar.number_input("Media poblacional", value=0.0, step=0.1)
    sigma = st.sidebar.number_input("Desvío estándar poblacional", min_value=0.1, value=1.0, step=0.1)
    p = None
elif tipo == "proporcion":
    p = st.sidebar.slider("Proporción poblacional", 0.01, 0.99, 0.5)
    mu, sigma = None, None

# --- Simulación ---
resultados = simular_intervalos(tipo, mu, sigma, n, nivel_confianza, num_simulaciones, p)

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(8, 6))

for i, (li, ls, est, contiene) in enumerate(resultados):
    color = "blue" if contiene else "red"
    ax.plot([li, ls], [i, i], color=color, marker="|")
    ax.plot(est, i, "o", color="black")

# Línea verde del valor poblacional
if tipo in ["media_varianza_conocida", "media_varianza_desconocida"]:
    ax.axvline(mu, color="green", linestyle="--")
    ax.set_title("Intervalos de confianza para la media")
elif tipo == "varianza":
    ax.axvline(sigma**2, color="green", linestyle="--")
    ax.set_title("Intervalos de confianza para la varianza")
elif tipo == "proporcion":
    ax.axvline(p, color="green", linestyle="--")
    ax.set_title("Intervalos de confianza para la proporción")

ax.set_xlabel("Valor")
ax.set_ylabel("Simulación")
st.pyplot(fig)

# --- Descargar CSV ---
df_resultados = pd.DataFrame(resultados, columns=["LI", "LS", "Estimador", "Contiene"])
csv = df_resultados.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Descargar simulaciones en CSV",
    data=csv,
    file_name="simulaciones.csv",
    mime="text/csv",
)
