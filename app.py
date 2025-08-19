import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stt
import altair as alt

# =========================
# Función para simular intervalos
# =========================
def simular_intervalos(tipo, n, sims, alpha, mu, sigma, p, rng):
    resultados = []
    for i in range(sims):
        if tipo == "Media con varianza conocida":
            muestra = rng.normal(mu, sigma, n)
            media_muestral = np.mean(muestra)
            z = stt.norm.ppf(1 - alpha/2)
            li = media_muestral - z * sigma/np.sqrt(n)
            ls = media_muestral + z * sigma/np.sqrt(n)
            valor_real = mu
            estimador = media_muestral

        elif tipo == "Media con varianza desconocida":
            muestra = rng.normal(mu, sigma, n)
            media_muestral = np.mean(muestra)
            s = np.std(muestra, ddof=1)
            t = stt.t.ppf(1 - alpha/2, df=n-1)
            li = media_muestral - t * s/np.sqrt(n)
            ls = media_muestral + t * s/np.sqrt(n)
            valor_real = mu
            estimador = media_muestral

        elif tipo == "Varianza":
            muestra = rng.normal(mu, sigma, n)
            s2 = np.var(muestra, ddof=1)
            chi2_inf = stt.chi2.ppf(1 - alpha/2, df=n-1)
            chi2_sup = stt.chi2.ppf(alpha/2, df=n-1)
            li = (n-1)*s2/chi2_inf
            ls = (n-1)*s2/chi2_sup
            valor_real = sigma**2
            estimador = s2

        elif tipo == "Proporción":
            muestra = rng.binomial(1, p, n)
            phat = np.mean(muestra)
            z = stt.norm.ppf(1 - alpha/2)
            li = phat - z*np.sqrt(phat*(1-phat)/n)
            ls = phat + z*np.sqrt(phat*(1-phat)/n)
            li = max(0, li)
            ls = min(1, ls)
            valor_real = p
            estimador = phat

        resultados.append([i, li, ls, estimador, valor_real])

    df = pd.DataFrame(resultados, columns=["Sim", "LI", "LS", "Estadístico", "Valor_real"])
    df["Cubre"] = (df["LI"] <= df["Valor_real"]) & (df["LS"] >= df["Valor_real"])
    return df

# =========================
# Función para graficar
# =========================
def plot_intervalos(df, valor_real, tipo, xmin, xmax):
    base = alt.Chart(df).encode(y=alt.Y("Sim:O", axis=None))

    intervalos = base.mark_rule(size=2).encode(
        x="LI:Q",
        x2="LS:Q",
        color=alt.condition("datum.Cubre", alt.value("steelblue"), alt.value("red"))
    )

    puntos = base.mark_point(filled=True, size=30).encode(
        x="Estadístico:Q",
        color=alt.condition("datum.Cubre", alt.value("steelblue"), alt.value("red")),
        tooltip=["LI", "LS", "Estadístico"]
    )

    linea_real = alt.Chart(pd.DataFrame({"valor": [valor_real]})).mark_rule(
        color="green", strokeWidth=2
    ).encode(
        x="valor:Q",
        tooltip=[alt.Tooltip("valor:Q", title=f"Valor poblacional ({tipo})")]
    )

    return (intervalos + puntos + linea_real).properties(
        width=700, height=400, title=f"Intervalos de Confianza para {tipo}"
    ).encode(
        x=alt.X("LI", scale=alt.Scale(domain=[xmin, xmax]))
    )

# =========================
# App principal
# =========================
st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")
st.title("Simulador de Intervalos de Confianza")

# Sidebar - parámetros
st.sidebar.header("Parámetros de la simulación")
tipo = st.sidebar.selectbox(
    "Tipo de intervalo",
    ["Media con varianza conocida", "Media con varianza desconocida", "Varianza", "Proporción"]
)

n = st.sidebar.slider("Tamaño muestral (n)", 2, 500, 30, 1)
sims = st.sidebar.slider("Número de simulaciones", 1, 200, 50, 1)
conf = st.sidebar.slider("Nivel de confianza (%)", 80, 99, 95, 1)
alpha = 1 - conf/100

# Inicializar estado
if "mu" not in st.session_state:
    st.session_state.mu = 0.0
if "sigma" not in st.session_state:
    st.session_state.sigma = 5.0
if "p" not in st.session_state:
    st.session_state.p = 0.5

# Sliders + inputs sincronizados
if tipo in ["Media con varianza conocida", "Media con varianza desconocida"]:
    col1, col2 = st.sidebar.columns([3,1])
    mu = col1.slider("Media poblacional (μ)", -100.0, 100.0, value=st.session_state.mu, step=0.1, key="mu_slider")
    mu = col2.number_input(" ", value=mu, step=0.1, key="mu_input")
    st.session_state.mu = mu

    col3, col4 = st.sidebar.columns([3,1])
    sigma = col3.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1, key="sigma_slider")
    sigma = col4.number_input("  ", value=sigma, step=0.1, key="sigma_input")
    st.session_state.sigma = sigma

elif tipo == "Varianza":
    col1, col2 = st.sidebar.columns([3,1])
    sigma = col1.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1, key="sigma_slider_var")
    sigma = col2.number_input(" ", value=sigma, step=0.1, key="sigma_input_var")
    st.session_state.sigma = sigma

elif tipo == "Proporción":
    col1, col2 = st.sidebar.columns([3,1])
    p = col1.slider("Proporción poblacional (p)", 0.01, 0.99, value=st.session_state.p, step=0.01, key="p_slider")
    p = col2.number_input("  ", value=p, step=0.01, key="p_input")
    st.session_state.p = p

# RNG
rng = np.random.default_rng(1234)

# Simulación
df = simular_intervalos(tipo, n, sims, alpha, st.session_state.mu, st.session_state.sigma, st.session_state.p, rng)
valor_real = df["Valor_real"].iloc[0]

# Ajuste de dominio (siempre incluye el valor poblacional)
xmin = min(df["LI"].min(), df["Estadístico"].min(), valor_real)
xmax = max(df["LS"].max(), df["Estadístico"].max(), valor_real)
span = xmax - xmin if xmax > 0 else 1
pad = span * 0.1
xmin_pad = min(xmin, valor_real) - pad
xmax_pad = max(xmax, valor_real) + pad

# Contenedor fijo arriba para el gráfico
chart_container = st.empty()
chart = plot_intervalos(df, valor_real, tipo, xmin_pad, xmax_pad)
chart_container.altair_chart(chart, use_container_width=True)

# Texto con porcentaje de cobertura
cubre_pct = df["Cubre"].mean() * 100
st.markdown(f"**Cobertura observada:** {cubre_pct:.1f}% de los intervalos contienen el valor real.")

