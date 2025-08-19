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

# Persistencia de parámetros con sliders visibles siempre
if "mu" not in st.session_state:
    st.session_state.mu = 0.0
if "sigma" not in st.session_state:
    st.session_state.sigma = 5.0
if "p" not in st.session_state:
    st.session_state.p = 0.5

if tipo in ["Media con varianza conocida", "Media con varianza desconocida"]:
    mu = st.sidebar.slider("Media poblacional (μ)", -100.0, 100.0, value=st.session_state.mu, step=0.1)
    sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1)
    st.session_state.mu = mu
    st.session_state.sigma = sigma
elif tipo == "Varianza":
    sigma = st.sidebar.slider("Desvío estándar poblacional (σ)", 0.1, 50.0, value=st.session_state.sigma, step=0.1)
    st.session_state.sigma = sigma
elif tipo == "Proporción":
    p = st.sidebar.slider("Proporción poblacional (p)", 0.01, 0.99, value=st.session_state.p, step=0.01)
    st.session_state.p = p

# RNG
rng = np.random.default_rng(1234)

# Simulaciones
if tipo == "Media con varianza conocida":
    muestras, medias, li, ls, contiene = sim_media_sigma_conocida(st.session_state.mu, st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.mu
elif tipo == "Media con varianza desconocida":
    muestras, medias, li, ls, contiene = sim_media_sigma_desconocida(st.session_state.mu, st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.mu
elif tipo == "Varianza":
    muestras, medias, li, ls, contiene = sim_varianza(st.session_state.sigma, n, sims, alpha, rng)
    valor_real = st.session_state.sigma**2
elif tipo == "Proporción":
    muestras, medias, li, ls, contiene = sim_proporcion(st.session_state.p, n, sims, alpha, rng)
    valor_real = st.session_state.p

# DataFrame
df = pd.DataFrame({
    "Simulación": np.arange(1, sims+1),
    "Estadístico": medias,
    "LI": li,
    "LS": ls,
    "Contiene": contiene
})

# Escala común para los gráficos
xmin = min(df["LI"].min(), valor_real)
xmax = max(df["LS"].max(), valor_real)

def plot_intervalos(df, valor_parametro, titulo):
    base = alt.Chart(df).encode(
        y=alt.Y("Simulación:O", sort="descending")
    )

    intervalos = base.mark_rule(size=2).encode(
        x=alt.X("LI", scale=alt.Scale(domain=[xmin, xmax])),
        x2="LS",
        color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
    )

    puntos = base.mark_point(filled=True, size=30).encode(
        x="Estadístico",
        color=alt.condition("datum.Contiene", alt.value("steelblue"), alt.value("red"))
    )

    linea = alt.Chart(pd.DataFrame({"valor":[valor_parametro]})).mark_rule(
        color="green", strokeDash=[4,4]
    ).encode(
        x=alt.X("valor", scale=alt.Scale(domain=[xmin, xmax]))
    )

    chart = (intervalos + puntos + linea).properties(
        width=700, height=400, 
        title=alt.TitleParams(titulo, orient="top")
    ).configure_view(
        continuousWidth=600, continuousHeight=400
    )
    return chart

# Gráfico
chart = plot_intervalos(df, valor_real, tipo)
st.altair_chart(chart, use_container_width=True)

# Tabla
primeras_20 = pd.DataFrame(muestras[:20])
primeras_20.index = np.arange(1, primeras_20.shape[0]+1)
primeras_20.columns = np.arange(1, primeras_20.shape[1]+1)
st.subheader("Primeros 20 conjuntos muestrales")
st.dataframe(primeras_20)
