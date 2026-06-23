"""
Menu interactivo para el sistema de prediccion de futbol internacional.

Uso:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.config import ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST
from src.prediction.poisson import predict_from_lambdas
from src.prediction.predictor import MatchPredictor

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 16


@st.cache_data(show_spinner="Buscando ciudad...", ttl=86400)
def geocode_city(city: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=5)
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        return results[0]["latitude"], results[0]["longitude"]
    except requests.RequestException:
        return None


@st.cache_data(show_spinner="Consultando pronostico...", ttl=3600)
def fetch_weather(city: str, date: str) -> dict | None:
    """Return forecast for *city* on *date* (ISO string), or None if unavailable."""
    days_ahead = (pd.Timestamp(date) - pd.Timestamp.now().normalize()).days
    if days_ahead < 0 or days_ahead > MAX_FORECAST_DAYS:
        return None
    coords = geocode_city(city)
    if coords is None:
        return None
    lat, lon = coords
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
                "timezone": "auto",
                "start_date": date,
                "end_date": date,
            },
            timeout=5,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily")
        if not daily or not daily.get("time"):
            return None
        return {
            "precip_prob": daily["precipitation_probability_max"][0],
            "temp_max": daily["temperature_2m_max"][0],
            "temp_min": daily["temperature_2m_min"][0],
            "wind": daily["windspeed_10m_max"][0],
        }
    except requests.RequestException:
        return None


def weather_adjustment(weather: dict) -> tuple[float, list[str]]:
    """Heuristic goal-expectancy multiplier based on adverse weather. Not learned by the model."""
    factor = 1.0
    notes = []
    if weather["precip_prob"] >= 70:
        factor *= 0.93
        notes.append(f"alta probabilidad de lluvia ({weather['precip_prob']:.0f}%)")
    if weather["temp_max"] >= 32:
        factor *= 0.95
        notes.append(f"calor extremo ({weather['temp_max']:.0f}°C)")
    if weather["temp_min"] <= 2:
        factor *= 0.96
        notes.append(f"frio extremo ({weather['temp_min']:.0f}°C)")
    if weather["wind"] >= 40:
        factor *= 0.97
        notes.append(f"viento fuerte ({weather['wind']:.0f} km/h)")
    return factor, notes

st.set_page_config(page_title="Predictor de Futbol Internacional", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def get_predictor(weight_bayes: float, weight_xgb: float) -> MatchPredictor:
    return MatchPredictor.load(weight_bayes=weight_bayes, weight_xgb=weight_xgb)


@st.cache_data(show_spinner="Cargando lista de selecciones...")
def get_team_list() -> list[str]:
    predictor = get_predictor(ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST)
    df = predictor._feature_df
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return teams


st.title("⚽ Predictor de Futbol Internacional")
st.caption("Ensemble Bayesiano (PyMC) + XGBoost Poisson sobre 49k+ partidos historicos (1872-presente)")

teams = get_team_list()

TOURNAMENT_OPTIONS = {
    "Amistoso": "Friendly",
    "Eliminatorias / Clasificacion": "Qualification",
    "Liga de Naciones": "Nations League",
    "Copa continental (Copa America / Eurocopa / etc.)": "Copa America",
    "Mundial": "FIFA World Cup",
    "Juegos Olimpicos": "Olympic Games",
    "Otro": "Other",
}

with st.sidebar:
    st.header("Configuracion del partido")
    home_team = st.selectbox("Equipo local", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    away_team = st.selectbox("Equipo visitante", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)
    tournament_label = st.selectbox(
        "Contexto del partido",
        list(TOURNAMENT_OPTIONS.keys()),
        help="El modelo aprendio del historico que el tipo de torneo afecta el resultado "
             "(ej. mundiales suelen ser mas cerrados que amistosos). Elegi la categoria real.",
    )
    tournament = TOURNAMENT_OPTIONS[tournament_label]
    is_neutral = st.checkbox("Cancha neutral")

    st.divider()
    st.subheader("Pesos del ensemble")
    weight_bayes = st.slider("Peso Bayesiano", 0.0, 1.0, ENSEMBLE_WEIGHT_BAYESIAN, 0.05)
    weight_xgb = round(1.0 - weight_bayes, 2)
    st.caption(f"Peso XGBoost: {weight_xgb}")

    st.divider()
    st.subheader("Clima (opcional)")
    use_weather = st.checkbox("Ajustar por clima del partido", value=False)
    match_city = ""
    match_date = pd.Timestamp.now().normalize()
    if use_weather:
        match_city = st.text_input("Ciudad de la sede", value="")
        match_date = pd.Timestamp(st.date_input("Fecha del partido", value=pd.Timestamp.now().date()))
        st.caption(
            "Pronostico real solo disponible hasta ~16 dias a futuro (Open-Meteo, gratis). "
            "El ajuste de goles esperados es una heuristica nuestra, no algo que el modelo aprendio de datos."
        )

    predict_clicked = st.button("Predecir", type="primary", use_container_width=True)

if not predict_clicked:
    st.info("Elegi dos selecciones en el menu lateral y presiona **Predecir**.")
    st.stop()

if home_team == away_team:
    st.error("Elegi dos selecciones distintas.")
    st.stop()

predictor = get_predictor(weight_bayes, weight_xgb)
pred = predictor.predict(
    home_team=home_team,
    away_team=away_team,
    is_neutral=is_neutral,
    tournament=tournament,
)

st.header(f"{pred.home_team} vs {pred.away_team}")
st.caption(f"Torneo: {tournament_label} | Cancha neutral: {is_neutral}")

if use_weather and match_city.strip():
    weather = fetch_weather(match_city.strip(), match_date.strftime("%Y-%m-%d"))
    if weather is None:
        st.warning(
            f"No se pudo obtener pronostico para '{match_city}' en esa fecha "
            f"(ciudad no encontrada o fuera del rango de {MAX_FORECAST_DAYS} dias). "
            "Se muestra la prediccion sin ajuste por clima."
        )
    else:
        factor, notes = weather_adjustment(weather)
        if notes:
            lh_adj = pred.expected_goals_home * factor
            la_adj = pred.expected_goals_away * factor
            poisson_adj = predict_from_lambdas(lh_adj, la_adj)
            st.subheader("🌦️ Ajuste por clima (heuristica, no aprendida por el modelo)")
            st.caption(
                f"{match_city} el {match_date.date()}: " + ", ".join(notes) +
                f". Factor aplicado a goles esperados: x{factor:.2f}"
            )
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.metric("xG local (sin ajuste → con ajuste)",
                          f"{lh_adj:.2f}", f"{lh_adj - pred.expected_goals_home:+.2f}")
            with wcol2:
                st.metric("xG visitante (sin ajuste → con ajuste)",
                          f"{la_adj:.2f}", f"{la_adj - pred.expected_goals_away:+.2f}")
            st.caption(
                f"1X2 ajustado: Local {poisson_adj.home_win:.1%} | Empate {poisson_adj.draw:.1%} | "
                f"Visitante {poisson_adj.away_win:.1%}  (original: {pred.home_win:.1%} / "
                f"{pred.draw:.1%} / {pred.away_win:.1%})"
            )
        else:
            st.caption(f"🌦️ Clima en {match_city} sin condiciones adversas relevantes — sin ajuste.")

# ── 1X2 + xG ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resultado del partido")
    result_df = pd.DataFrame({
        "Resultado": [f"Victoria {pred.home_team}", "Empate", f"Victoria {pred.away_team}"],
        "Probabilidad": [pred.home_win, pred.draw, pred.away_win],
    })
    fig = px.bar(result_df, x="Resultado", y="Probabilidad", text_auto=".1%",
                 color="Resultado", color_discrete_sequence=["#2E86AB", "#A8A8A8", "#E63946"])
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Goles esperados (xG)")
    xg_df = pd.DataFrame({
        "Equipo": [pred.home_team, pred.away_team],
        "xG": [pred.expected_goals_home, pred.expected_goals_away],
    })
    fig = px.bar(xg_df, x="Equipo", y="xG", text_auto=".2f", color="Equipo",
                 color_discrete_sequence=["#2E86AB", "#E63946"])
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Marcador mas probable", pred.most_likely_score)

# ── Mercados de goles ─────────────────────────────────────────────────────────
st.subheader("Mercados de goles")
markets_df = pd.DataFrame({
    "Mercado": ["BTTS", "Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5"],
    "Probabilidad": [pred.btts, pred.over_0_5, pred.over_1_5, pred.over_2_5, pred.over_3_5],
})
fig = px.bar(markets_df, x="Mercado", y="Probabilidad", text_auto=".1%", color="Probabilidad",
             color_continuous_scale="Blues")
fig.update_layout(yaxis_tickformat=".0%", height=320, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# ── Distribucion de marcadores (heatmap) ──────────────────────────────────────
st.subheader("Distribucion de marcadores")
scores = [(int(k.split("-")[0]), int(k.split("-")[1]), v) for k, v in pred.score_probabilities.items()]
max_goals = max(max(h, a) for h, a, _ in scores)
matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
for h, a, p in scores:
    matrix[h][a] = p
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=[str(i) for i in range(max_goals + 1)],
    y=[str(i) for i in range(max_goals + 1)],
    colorscale="Blues",
    texttemplate="%{z:.1%}",
    hovertemplate=f"{pred.home_team}: %{{y}}<br>{pred.away_team}: %{{x}}<br>Prob: %{{z:.1%}}<extra></extra>",
))
fig.update_layout(
    xaxis_title=f"Goles {pred.away_team}", yaxis_title=f"Goles {pred.home_team}",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ── Eventos del partido ───────────────────────────────────────────────────────
e = pred.events
st.subheader("Posesion, tiros y disciplina (estimado)")
col1, col2, col3 = st.columns(3)

with col1:
    fig = go.Figure(data=[go.Pie(
        labels=[pred.home_team, pred.away_team],
        values=[e.home_possession, e.away_possession],
        hole=0.5,
        marker_colors=["#2E86AB", "#E63946"],
    )])
    fig.update_layout(title="Posesion", height=320)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    shots_df = pd.DataFrame({
        "Equipo": [pred.home_team, pred.away_team, pred.home_team, pred.away_team],
        "Tipo": ["Tiros totales", "Tiros totales", "A puerta", "A puerta"],
        "Valor": [e.home_shots, e.away_shots, e.home_shots_on_target, e.away_shots_on_target],
    })
    fig = px.bar(shots_df, x="Tipo", y="Valor", color="Equipo", barmode="group",
                 color_discrete_sequence=["#2E86AB", "#E63946"])
    fig.update_layout(title="Tiros", height=320)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    cards_df = pd.DataFrame({
        "Equipo": [pred.home_team, pred.away_team],
        "Amarillas esperadas": [e.home_yellow_cards, e.away_yellow_cards],
    })
    fig = px.bar(cards_df, x="Equipo", y="Amarillas esperadas", text_auto=".2f", color="Equipo",
                 color_discrete_sequence=["#FFD60A", "#FFA94D"])
    fig.update_layout(title="Tarjetas amarillas", height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Corners esperados: {e.home_corners:.1f} - {e.away_corners:.1f}  |  "
    f"P(roja) {pred.home_team}: {e.home_red_card_prob:.1%}  |  "
    f"P(roja) {pred.away_team}: {e.away_red_card_prob:.1%}"
)

# ── Patrones historicos ───────────────────────────────────────────────────────
if pred.patterns is not None and pred.patterns.n_similar_matches >= 10:
    st.subheader("Patrones historicos similares")
    pat = pred.patterns
    st.caption(f"{pat.elo_bucket_label} — {pat.n_similar_matches} partidos similares encontrados")
    strong = [p for p in pat.patterns if p.strength in ("FUERTE", "MODERADO")]
    if strong:
        pat_df = pd.DataFrame({
            "Patron": [p.description for p in strong],
            "Frecuencia": [p.frequency for p in strong],
            "Fuerza": [p.strength for p in strong],
        }).sort_values("Frecuencia", ascending=True)
        fig = px.bar(pat_df, x="Frecuencia", y="Patron", color="Fuerza", orientation="h",
                     text_auto=".1%", color_discrete_map={"FUERTE": "#2E86AB", "MODERADO": "#A8A8A8"})
        fig.update_layout(xaxis_tickformat=".0%", height=300)
        st.plotly_chart(fig, use_container_width=True)

# ── Detalle del modelo ────────────────────────────────────────────────────────
with st.expander("Detalle del modelo (Bayesiano vs XGBoost)"):
    detail_df = pd.DataFrame({
        "Modelo": ["Bayesiano", "XGBoost", "Ensemble"],
        "xG local": [pred.lambda_bayes_home, pred.lambda_xgb_home, pred.expected_goals_home],
        "xG visitante": [pred.lambda_bayes_away, pred.lambda_xgb_away, pred.expected_goals_away],
        "Peso": [f"{pred.model_weights['bayesian']:.0%}", f"{pred.model_weights['xgboost']:.0%}", "100%"],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    st.caption("Tiros, corners y tarjetas son estimaciones estadisticas basadas en xG, no datos reales.")
