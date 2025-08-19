# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm, t as tdist, chi2

st.set_page_config(page_title="Simulador de Intervalos de Confianza", layout="wide")

# =========================================================
# Utilidades de estado y simulación (persistencia)
# =========================================================
def ensure_state(key):
    if key not in st.session_state:
        st.session_state[key] = {
            "n": None,
            "reps": 0,
            "samples": [],     # lista de np.arrays con los datos muestrales
            "meta": {},        # info de cuándo se generaron (μ, σ, dist, etc.)
        }

def adjust_simulations(state_key, desired_reps, n, make_sample_fn, meta_now):
    """
    Ajusta el número de simulaciones persistentes:
    - Si cambia n -> regenera todo.
    - Si desired_reps > existentes -> agrega muestras nuevas.
    - Si desired_reps < existentes -> recorta.
    Nota: solo se llama cuando cambian n o reps; cambiar confianza NO toca las muestras.
    """
    s = st.session_state[state_key]
    need_regen = (s["n"] is None) or (s["n"] != n)

    if need_regen:
        s["samples"] = [make_sample_fn(n) for _ in range(desired_reps)]
        s["n"] = n
        s["reps"] = desired_reps
        s["meta"] = meta_now.copy()
        return

    # mismo n: agregar o recortar sin perder lo existente
    cur = len(s["samples"])
    if desired_reps > cur:
        s["samples"].extend(make_sample_fn(n) for _ in range(desired_reps - cur))
    elif desired_reps < cur:
        s["samples"] = s["samples"][:desired_reps]
    s["reps"] = desired_reps
    # mantenemos meta del momento en que se (re)generaron las muestras

def x_domain_with_padding(li_vals, ls_vals, pad_ratio=0.1):
    xmin = float(np.nanmin(li_vals))
    xmax = float(np.nanmax(ls_vals))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None
    if xmax == xmin:
        # rango degenerado: expandimos un poco
        rng = 1.0 if xmax == 0 else abs(xmax) * 0.1
        return [xmin - rng, xmax + rng]
    rng = xmax - xmin
    pad = rng * pad_ratio
    return [xmin - pad, xmax + pad]

# =========================================================
# Generadores de muestras
# =========================================================
def make_sampler_media(mu, sigma, dist_name, df_t=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    if dist_name == "Normal":
        def _f(n):
            return rng.normal(mu, sigma, n)
        return _f

    if dist_name == "t-Student":
        df_eff = 8 if (df_t is None or df_t <= 2) else df_t
        # Estandar t con var=df/(df-2); reescalamos para tener desvío ~sigma
        scale = sigma / np.sqrt(df_eff / (df_eff - 2))
        def _f(n):
            return mu + scale * rng.standard_t(df_eff, size=n)
        return _f

    # Lognormal (requiere μ>0 para definir parámetros m,s de la lognormal)
    def _f(n):
        mu_eff = mu if mu > 0 else max(1e-6, abs(mu))
        s2 = np.log(1 + (sigma**2) / (mu_eff**2))
        s = np.sqrt(s2)
        m = np.log(mu_eff) - s2/2
        return rng.lognormal(mean=m, sigma=s, size=n)
    return _f

def make_sampler_varianza(mu, sigma, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    def _f(n):
        # Para IC de varianza clásico asumimos población Normal
        return rng.normal(mu, sigma, n)
    return _f

def make_sampler_proporcion(p, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    def _f(n):
        return rng.binomial(1, p, n)
    return _f

# =========================================================
# Cálculo de intervalos
# =========================================================
def ic_media_sigma_conocida(x, sigma, alpha):
    xbar = np.mean(x)
    z = norm.ppf(1 - alpha/2)
    se = sigma / np.sqrt(len(x))
    return xbar - z*se, xbar + z*se, xbar

def ic_media_sigma_desconocida(x, alpha):
    xbar = np.mean(x)
    s = np.std(x, ddof=1)
    tcrit = tdist.ppf(1 - alpha/2, df=len(x)-1)
    se = s / np.sqrt(len(x))
    return xbar - tcrit*se, xbar + tcrit*se, xbar

def ic_varianza(x, alpha):
    s2 = np.var(x, ddof=1)
    df = len(x) - 1
    lo = df * s2 / chi2.ppf(1 - alpha/2, df=df)
    hi = df * s2 / chi2.ppf(alpha/2, df=df)
    return lo, hi, s2

def ic_proporcion_wald(x, alpha):
    # Wald clásico (pediste lineal y simple; si querés luego cambiamos a Wilson)
    phat = np.mean(x)
    z = norm.ppf(1 - alpha/2)
    se = np.sqrt(max(phat*(1 - phat), 1e-12)/len(x))
    lo = max(0.0, phat - z*se)
    hi = min(1.0, phat + z*se)
    return lo, hi, phat

# =========================================================
# UI
# =========================================================
st.title("📊 Simulador de Intervalos de Confianza")
st.caption(
    "Los datos muestrales **persisten**: solo se regeneran si cambiás el **tamaño muestral (n)** "
    "o el **número de simulaciones**. Cambiar el **nivel de confianza** recalcula los IC sobre "
    "las mismas muestras, para comparar longitud vs. confianza."
)

tabs = st.tabs([
    "Media (σ conocida)",
    "Media (σ desconocida)",
    "Varianza",
    "Proporción",
])

# -------------------------------
# TAB 1: Media (σ conocida)
# -------------------------------
with tabs[0]:
    ensure_state("state_mean_z")

    colL, colR = st.columns([2, 1])
    with colL:
        st.subheader("Intervalo de confianza para la media (σ conocida)")
        n_z = st.slider("Tamaño muestral (n)", 5, 2000, 30, key="n_z")
        reps_z = st.slider("Número de simulaciones", 10, 1000, 100, step=10, key="reps_z")
        conf_z = st.slider("Nivel de confianza (%)", 80, 99, 95, key="conf_z")/100.0
        alpha_z = 1 - conf_z
    with colR:
        mu_z = st.number_input("Media poblacional (μ)", value=0.0, key="mu_z")
        sigma_z = st.number_input("Desvío estándar poblacional (σ)", value=1.0, min_value=0.0001, key="sigma_z")
        dist_z = st.selectbox("Distribución poblacional", ["Normal", "t-Student", "Lognormal"], key="dist_z")
        df_z = None
        if dist_z == "t-Student":
            df_z = st.slider("Grados de libertad (t)", 3, 100, 8, key="df_z")

    # Ajuste/creación de simulaciones solo si cambian n o reps
    meta_z = {"mu": mu_z, "sigma": sigma_z, "dist": dist_z, "df": df_z}
    sampler_z = make_sampler_media(mu_z, sigma_z, dist_z, df_t=df_z)
    adjust_simulations("state_mean_z", reps_z, n_z, sampler_z, meta_z)

    # Calcular IC sobre las muestras persistentes
    s_z = st.session_state["state_mean_z"]
    recs = []
    for i, x in enumerate(s_z["samples"], 1):
        li, ls, est = ic_media_sigma_conocida(x, sigma_z, alpha_z)
        recs.append((i, li, ls, est))
    df_z = pd.DataFrame(recs, columns=["Sim", "LI", "LS", "Estimador"])
    covers = (df_z["LI"] <= mu_z) & (mu_z <= df_z["LS"])
    coverage = covers.mean()*100

    # Gráfico (escala lineal auto-ajustada)
    domain = x_domain_with_padding(df_z["LI"].values, df_z["LS"].values, pad_ratio=0.1)
    base = alt.Chart(df_z).mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=domain), title="Valor"),
        x2="LS:Q",
        y=alt.Y("Sim:O", sort="descending", title=None),
        color=alt.condition(alt.datum.LI <= mu_z, alt.value("black"), alt.value("black"))
    )
    # color por cobertura (negro si cubre, rojo si no)
    base = alt.Chart(df_z.assign(Cubre=covers)).mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=domain), title="Valor"),
        x2="LS:Q",
        y=alt.Y("Sim:O", sort="descending", title=None),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    vline = alt.Chart(pd.DataFrame({"v":[mu_z]})).mark_rule(strokeDash=[6,4], color="steelblue").encode(x="v:Q")
    point = alt.Chart(df_z.assign(Cubre=covers)).mark_point(size=28, filled=True).encode(
        x="Estimador:Q", y=alt.Y("Sim:O", sort="descending"),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    st.altair_chart((base + vline + point).properties(height=520).configure_view(strokeWidth=0), use_container_width=True)

    st.metric("Cobertura observada", f"{coverage:.1f}%")
    st.caption(f"Muestras generadas con {s_z['meta']}. Cambiá n o #simulaciones para regenerar.")

    # Tabla y descarga
    def fmt_arr(a):
        return "[" + ", ".join(f"{v:.4g}" for v in a) + "]"
    preview_z = pd.DataFrame({
        "Simulación": df_z["Sim"].head(20),
        "Datos muestrales": [fmt_arr(a) for a in s_z["samples"][:20]],
        "Estimador": df_z["Estimador"].head(20),
        "LI": df_z["LI"].head(20),
        "LS": df_z["LS"].head(20),
        "Cubre": covers.head(20),
    })
    st.subheader("📊 Primeros 20 conjuntos muestrales")
    st.dataframe(preview_z, use_container_width=True)
    # CSV completo
    flat_df_z = pd.DataFrame({
        "Simulación": df_z["Sim"],
        "Datos muestrales": [fmt_arr(a) for a in s_z["samples"]],
        "Estimador": df_z["Estimador"],
        "LI": df_z["LI"],
        "LS": df_z["LS"],
        "Cubre": covers,
    })
    st.download_button("⬇️ Descargar CSV (media, σ conocida)", data=flat_df_z.to_csv(index=False).encode("utf-8"),
                       file_name="ic_media_sigma_conocida.csv", mime="text/csv")

# -------------------------------
# TAB 2: Media (σ desconocida)
# -------------------------------
with tabs[1]:
    ensure_state("state_mean_t")

    colL, colR = st.columns([2, 1])
    with colL:
        st.subheader("Intervalo de confianza para la media (σ desconocida)")
        n_t = st.slider("Tamaño muestral (n)", 5, 2000, 30, key="n_t")
        reps_t = st.slider("Número de simulaciones", 10, 1000, 100, step=10, key="reps_t")
        conf_t = st.slider("Nivel de confianza (%)", 80, 99, 95, key="conf_t")/100.0
        alpha_t = 1 - conf_t
    with colR:
        mu_t = st.number_input("Media poblacional (μ)", value=0.0, key="mu_t")
        sigma_t = st.number_input("Desvío estándar poblacional (σ)", value=1.0, min_value=0.0001, key="sigma_t")
        dist_t = st.selectbox("Distribución poblacional", ["Normal", "t-Student", "Lognormal"], key="dist_t")
        df_t = None
        if dist_t == "t-Student":
            df_t = st.slider("Grados de libertad (t)", 3, 100, 8, key="df_t2")

    meta_t = {"mu": mu_t, "sigma": sigma_t, "dist": dist_t, "df": df_t}
    sampler_t = make_sampler_media(mu_t, sigma_t, dist_t, df_t=df_t)
    adjust_simulations("state_mean_t", reps_t, n_t, sampler_t, meta_t)

    s_t = st.session_state["state_mean_t"]
    recs = []
    for i, x in enumerate(s_t["samples"], 1):
        li, ls, est = ic_media_sigma_desconocida(x, alpha_t)
        recs.append((i, li, ls, est))
    df_tdf = pd.DataFrame(recs, columns=["Sim", "LI", "LS", "Estimador"])
    covers_t = (df_tdf["LI"] <= mu_t) & (mu_t <= df_tdf["LS"])
    coverage_t = covers_t.mean()*100

    domain_t = x_domain_with_padding(df_tdf["LI"].values, df_tdf["LS"].values, pad_ratio=0.1)
    base_t = alt.Chart(df_tdf.assign(Cubre=covers_t)).mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=domain_t), title="Valor"),
        x2="LS:Q",
        y=alt.Y("Sim:O", sort="descending", title=None),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    vline_t = alt.Chart(pd.DataFrame({"v":[mu_t]})).mark_rule(strokeDash=[6,4], color="steelblue").encode(x="v:Q")
    point_t = alt.Chart(df_tdf.assign(Cubre=covers_t)).mark_point(size=28, filled=True).encode(
        x="Estimador:Q", y=alt.Y("Sim:O", sort="descending"),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    st.altair_chart((base_t + vline_t + point_t).properties(height=520).configure_view(strokeWidth=0), use_container_width=True)

    st.metric("Cobertura observada", f"{coverage_t:.1f}%")
    st.caption(f"Muestras generadas con {s_t['meta']}. Cambiá n o #simulaciones para regenerar.")

    def fmt_arr(a): return "[" + ", ".join(f"{v:.4g}" for v in a) + "]"
    preview_t = pd.DataFrame({
        "Simulación": df_tdf["Sim"].head(20),
        "Datos muestrales": [fmt_arr(a) for a in s_t["samples"][:20]],
        "Estimador": df_tdf["Estimador"].head(20),
        "LI": df_tdf["LI"].head(20),
        "LS": df_tdf["LS"].head(20),
        "Cubre": covers_t.head(20),
    })
    st.subheader("📊 Primeros 20 conjuntos muestrales")
    st.dataframe(preview_t, use_container_width=True)

    flat_df_t = pd.DataFrame({
        "Simulación": df_tdf["Sim"],
        "Datos muestrales": [fmt_arr(a) for a in s_t["samples"]],
        "Estimador": df_tdf["Estimador"],
        "LI": df_tdf["LI"],
        "LS": df_tdf["LS"],
        "Cubre": covers_t,
    })
    st.download_button("⬇️ Descargar CSV (media, σ desconocida)", data=flat_df_t.to_csv(index=False).encode("utf-8"),
                       file_name="ic_media_sigma_desconocida.csv", mime="text/csv")

# -------------------------------
# TAB 3: Varianza (χ²)
# -------------------------------
with tabs[2]:
    ensure_state("state_var")

    colL, colR = st.columns([2, 1])
    with colL:
        st.subheader("Intervalo de confianza para la varianza (χ²)")
        n_v = st.slider("Tamaño muestral (n)", 5, 2000, 30, key="n_v")
        reps_v = st.slider("Número de simulaciones", 10, 1000, 100, step=10, key="reps_v")
        conf_v = st.slider("Nivel de confianza (%)", 80, 99, 95, key="conf_v")/100.0
        alpha_v = 1 - conf_v
    with colR:
        mu_v = st.number_input("Media poblacional (μ)", value=0.0, key="mu_v")
        sigma_v = st.number_input("Desvío estándar poblacional (σ)", value=1.0, min_value=0.0001, key="sigma_v")

    meta_v = {"mu": mu_v, "sigma": sigma_v, "dist": "Normal"}
    sampler_v = make_sampler_varianza(mu_v, sigma_v)
    adjust_simulations("state_var", reps_v, n_v, sampler_v, meta_v)

    s_v = st.session_state["state_var"]
    recs = []
    for i, x in enumerate(s_v["samples"], 1):
        li, ls, est = ic_varianza(x, alpha_v)
        recs.append((i, li, ls, est))
    df_v = pd.DataFrame(recs, columns=["Sim", "LI", "LS", "Estimador"])
    covers_v = (df_v["LI"] <= sigma_v**2) & (sigma_v**2 <= df_v["LS"])
    coverage_v = covers_v.mean()*100

    domain_v = x_domain_with_padding(df_v["LI"].values, df_v["LS"].values, pad_ratio=0.1)
    base_v = alt.Chart(df_v.assign(Cubre=covers_v)).mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=domain_v), title="Valor (σ²)"),
        x2="LS:Q",
        y=alt.Y("Sim:O", sort="descending", title=None),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    vline_v = alt.Chart(pd.DataFrame({"v":[sigma_v**2]})).mark_rule(strokeDash=[6,4], color="steelblue").encode(x="v:Q")
    point_v = alt.Chart(df_v.assign(Cubre=covers_v)).mark_point(size=28, filled=True).encode(
        x="Estimador:Q", y=alt.Y("Sim:O", sort="descending"),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    st.altair_chart((base_v + vline_v + point_v).properties(height=520).configure_view(strokeWidth=0), use_container_width=True)

    st.metric("Cobertura observada", f"{coverage_v:.1f}%")
    st.caption(f"Muestras generadas con {s_v['meta']}. Cambiá n o #simulaciones para regenerar. (IC de varianza asume Normalidad)")

    def fmt_arr(a): return "[" + ", ".join(f"{v:.4g}" for v in a) + "]"
    preview_v = pd.DataFrame({
        "Simulación": df_v["Sim"].head(20),
        "Datos muestrales": [fmt_arr(a) for a in s_v["samples"][:20]],
        "Estimador (s²)": df_v["Estimador"].head(20),
        "LI": df_v["LI"].head(20),
        "LS": df_v["LS"].head(20),
        "Cubre": covers_v.head(20),
    })
    st.subheader("📊 Primeros 20 conjuntos muestrales")
    st.dataframe(preview_v, use_container_width=True)

    flat_df_v = pd.DataFrame({
        "Simulación": df_v["Sim"],
        "Datos muestrales": [fmt_arr(a) for a in s_v["samples"]],
        "Estimador (s²)": df_v["Estimador"],
        "LI": df_v["LI"],
        "LS": df_v["LS"],
        "Cubre": covers_v,
    })
    st.download_button("⬇️ Descargar CSV (varianza)", data=flat_df_v.to_csv(index=False).encode("utf-8"),
                       file_name="ic_varianza.csv", mime="text/csv")

# -------------------------------
# TAB 4: Proporción
# -------------------------------
with tabs[3]:
    ensure_state("state_prop")

    colL, colR = st.columns([2, 1])
    with colL:
        st.subheader("Intervalo de confianza para la proporción")
        n_p = st.slider("Tamaño muestral (n)", 5, 2000, 50, key="n_p")
        reps_p = st.slider("Número de simulaciones", 10, 1000, 100, step=10, key="reps_p")
        conf_p = st.slider("Nivel de confianza (%)", 80, 99, 95, key="conf_p")/100.0
        alpha_p = 1 - conf_p
    with colR:
        p_p = st.slider("Proporción poblacional (p)", 0.01, 0.99, 0.5, 0.01, key="p_p")

    meta_p = {"p": p_p}
    sampler_p = make_sampler_proporcion(p_p)
    adjust_simulations("state_prop", reps_p, n_p, sampler_p, meta_p)

    s_p = st.session_state["state_prop"]
    recs = []
    for i, x in enumerate(s_p["samples"], 1):
        li, ls, est = ic_proporcion_wald(x, alpha_p)
        recs.append((i, li, ls, est))
    df_p = pd.DataFrame(recs, columns=["Sim", "LI", "LS", "Estimador"])
    covers_p = (df_p["LI"] <= p_p) & (p_p <= df_p["LS"])
    coverage_p = covers_p.mean()*100

    domain_p = x_domain_with_padding(df_p["LI"].values, df_p["LS"].values, pad_ratio=0.1)
    base_p = alt.Chart(df_p.assign(Cubre=covers_p)).mark_rule(size=2).encode(
        x=alt.X("LI:Q", scale=alt.Scale(domain=domain_p), title="Proporción"),
        x2="LS:Q",
        y=alt.Y("Sim:O", sort="descending", title=None),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    vline_p = alt.Chart(pd.DataFrame({"v":[p_p]})).mark_rule(strokeDash=[6,4], color="steelblue").encode(x="v:Q")
    point_p = alt.Chart(df_p.assign(Cubre=covers_p)).mark_point(size=28, filled=True).encode(
        x="Estimador:Q", y=alt.Y("Sim:O", sort="descending"),
        color=alt.condition("datum.Cubre", alt.value("black"), alt.value("red"))
    )
    st.altair_chart((base_p + vline_p + point_p).properties(height=520).configure_view(strokeWidth=0), use_container_width=True)

    st.metric("Cobertura observada", f"{coverage_p:.1f}%")
    st.caption(f"Muestras generadas con {s_p['meta']}. Cambiá n o #simulaciones para regenerar.")

    def fmt_arr(a): return "[" + ", ".join(str(int(v)) for v in a) + "]"
    preview_p = pd.DataFrame({
        "Simulación": df_p["Sim"].head(20),
        "Datos muestrales (0/1)": [fmt_arr(a) for a in s_p["samples"][:20]],
        "Estimador (p̂)": df_p["Estimador"].head(20),
        "LI": df_p["LI"].head(20),
        "LS": df_p["LS"].head(20),
        "Cubre": covers_p.head(20),
    })
    st.subheader("📊 Primeros 20 conjuntos muestrales")
    st.dataframe(preview_p, use_container_width=True)

    flat_df_p = pd.DataFrame({
        "Simulación": df_p["Sim"],
        "Datos muestrales (0/1)": [fmt_arr(a) for a in s_p["samples"]],
        "Estimador (p̂)": df_p["Estimador"],
        "LI": df_p["LI"],
        "LS": df_p["LS"],
        "Cubre": covers_p,
    })
    st.download_button("⬇️ Descargar CSV (proporción)", data=flat_df_p.to_csv(index=False).encode("utf-8"),
                       file_name="ic_proporcion.csv", mime="text/csv")
