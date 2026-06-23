"""
Menu interactivo para el sistema de prediccion de futbol internacional.

Uso:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import io

import matplotlib.font_manager as fm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.config import ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST
from src.prediction.poisson import predict_from_lambdas
from src.prediction.predictor import MatchPredictor, MatchPrediction

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 16

# ── Paleta de colores consistente (estetica 7a0: crema/rojo-naranja/dorado) ───
# Tonos pastel con borde oscuro, en linea con el estilo grafico de la pagina de referencia.
HOME_COLOR = "#9DC9A4"
AWAY_COLOR = "#F2A28E"
DRAW_COLOR = "#E8CB8C"
CARD_COLOR = "#F2D399"
CARD_COLOR_2 = "#F0B88B"
ACCENT_SCALE = [[0.0, "#FBF3E3"], [0.5, "#E8CB8C"], [1.0, "#D98B4B"]]
CREAM_BG = "#F3ECD8"
INK_TEXT = "#1B1A17"
BORDER_WIDTH = 1.6


def _style_borders(fig, width: float = BORDER_WIDTH) -> None:
    """Apply a dark outline to bars/markers/pie slices, matching the 7a0 look."""
    fig.update_traces(marker=dict(line=dict(color=INK_TEXT, width=width)), selector=dict(type="bar"))
    fig.update_traces(marker=dict(line=dict(color=INK_TEXT, width=width)), selector=dict(type="pie"))

# ── Banderas ───────────────────────────────────────────────────────────────────
FLAG_OVERRIDES = {
    "USA": "US", "United States": "US", "South Korea": "KR", "North Korea": "KP",
    "Ivory Coast": "CI", "DR Congo": "CD", "Congo": "CG", "Cape Verde": "CV",
    "England": "GB-ENG", "Scotland": "GB-SCT", "Wales": "GB-WLS",
    "Northern Ireland": "GB-NIR", "Republic of Ireland": "IE", "Russia": "RU",
    "Czech Republic": "CZ", "Iran": "IR", "Bolivia": "BO", "Venezuela": "VE",
    "Tanzania": "TZ", "Vietnam": "VN", "Syria": "SY", "Laos": "LA",
    "Brunei": "BN", "Moldova": "MD", "North Macedonia": "MK", "Macedonia": "MK",
    "Eswatini": "SZ", "Curacao": "CW", "Bosnia and Herzegovina": "BA",
}


def _flag_from_alpha2(code: str) -> str:
    if len(code) != 2 or not code.isalpha():
        return "🏳️"
    return chr(0x1F1E6 + ord(code[0].upper()) - ord("A")) + chr(0x1F1E6 + ord(code[1].upper()) - ord("A"))


@st.cache_data
def team_flag(team: str) -> str:
    code = FLAG_OVERRIDES.get(team)
    if code is None:
        try:
            code = pycountry.countries.lookup(team).alpha_2
        except LookupError:
            return "🏳️"
    if "-" in code:
        return "🏴"
    return _flag_from_alpha2(code)


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


def build_summary_image(pred: MatchPrediction, tournament_label: str) -> bytes:
    """Render a shareable PNG summary card for the prediction."""
    W, H = 900, 540
    bg = (243, 236, 216)
    ink = (27, 26, 23)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    bold_path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    reg_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    f_title = ImageFont.truetype(bold_path, 32)
    f_sub = ImageFont.truetype(reg_path, 18)
    f_label = ImageFont.truetype(reg_path, 17)
    f_big = ImageFont.truetype(bold_path, 26)
    f_footer = ImageFont.truetype(reg_path, 14)

    draw.text((40, 30), f"{pred.home_team} vs {pred.away_team}", font=f_title, fill=ink)
    draw.text((40, 75), tournament_label, font=f_sub, fill=(110, 104, 90))

    bars = [
        (f"Gana {pred.home_team}", pred.home_win, (157, 201, 164)),
        ("Empate", pred.draw, (232, 203, 140)),
        (f"Gana {pred.away_team}", pred.away_win, (242, 162, 142)),
    ]
    bar_y = 130
    bar_width = 740
    for label, prob, color in bars:
        draw.text((40, bar_y), label, font=f_label, fill=ink)
        draw.text((40 + bar_width - 60, bar_y), f"{prob:.1%}", font=f_label, fill=ink)
        bar_top = bar_y + 24
        draw.rectangle([40, bar_top, 40 + bar_width, bar_top + 20], fill=(225, 216, 192), outline=ink, width=2)
        draw.rectangle([40, bar_top, 40 + max(int(bar_width * prob), 2), bar_top + 20], fill=color, outline=ink, width=2)
        bar_y += 65

    y2 = bar_y + 15
    draw.text(
        (40, y2),
        f"xG {pred.home_team}: {pred.expected_goals_home:.2f}    "
        f"xG {pred.away_team}: {pred.expected_goals_away:.2f}",
        font=f_label, fill=ink,
    )
    draw.text((40, y2 + 35), f"Marcador mas probable: {pred.most_likely_score}", font=f_big, fill=ink)
    draw.text(
        (40, y2 + 80),
        f"BTTS {pred.btts:.0%}   Over 1.5 {pred.over_1_5:.0%}   Over 2.5 {pred.over_2_5:.0%}",
        font=f_label, fill=(90, 84, 72),
    )

    draw.text((40, H - 35), "Generado con Predictor de Futbol Internacional", font=f_footer, fill=(140, 132, 114))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


st.set_page_config(page_title="Predictor de Futbol Internacional", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Hanken Grotesk', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Anton', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Anton', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Hanken Grotesk', sans-serif !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    div.stButton > button, div.stDownloadButton > button {
        font-family: 'Anton', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-radius: 2px;
        background-color: #E8462B;
        color: #FFFFFF;
        border: none;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #C93A22;
        color: #FFFFFF;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-family: 'Hanken Grotesk', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        background-size: contain;
        background-repeat: no-repeat;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:nth-child(1)::before {
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUIxQTE3IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGxpbmUgeDE9IjE4IiB5MT0iMjAiIHgyPSIxOCIgeTI9IjEwIi8+PGxpbmUgeDE9IjEyIiB5MT0iMjAiIHgyPSIxMiIgeTI9IjQiLz48bGluZSB4MT0iNiIgeTE9IjIwIiB4Mj0iNiIgeTI9IjE0Ii8+PC9zdmc+");
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:nth-child(2)::before {
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUIxQTE3IiBzdHJva2Utd2lkdGg9IjIiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjkiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI0LjUiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxIiBmaWxsPSIjMUIxQTE3Ii8+PC9zdmc+");
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:nth-child(3)::before {
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUIxQTE3IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMjIgMTIgMTggMTIgMTUgMjEgOSAzIDYgMTIgMiAxMiIvPjwvc3ZnPg==");
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:nth-child(4)::before {
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUIxQTE3IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iOSIvPjxwb2x5bGluZSBwb2ludHM9IjEyIDcgMTIgMTIgMTYgMTQiLz48L3N2Zz4=");
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:nth-child(5)::before {
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMUIxQTE3IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTEiIGN5PSIxMSIgcj0iNyIvPjxsaW5lIHgxPSIyMSIgeTE9IjIxIiB4Mj0iMTYuNjUiIHkyPSIxNi42NSIvPjwvc3ZnPg==");
    }

    div.stDownloadButton > button p::before {
        content: "";
        display: inline-block;
        width: 14px;
        height: 14px;
        background-size: contain;
        background-repeat: no-repeat;
        margin-right: 6px;
        vertical-align: middle;
        background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjRkZGRkZGIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTIxIDE1djRhMiAyIDAgMCAxLTIgMkg1YTIgMiAwIDAgMS0yLTJ2LTQiLz48cG9seWxpbmUgcG9pbnRzPSI3IDEwIDEyIDE1IDE3IDEwIi8+PGxpbmUgeDE9IjEyIiB5MT0iMTUiIHgyPSIxMiIgeTI9IjMiLz48L3N2Zz4=");
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E3DCC8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def get_predictor(weight_bayes: float, weight_xgb: float) -> MatchPredictor:
    return MatchPredictor.load(weight_bayes=weight_bayes, weight_xgb=weight_xgb)


@st.cache_data(show_spinner="Cargando lista de selecciones...")
def get_team_list() -> list[str]:
    predictor = get_predictor(ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST)
    df = predictor._feature_df
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return teams


@st.cache_data
def get_h2h_matches(_df: pd.DataFrame, home: str, away: str, n: int = 10) -> pd.DataFrame:
    """Most recent *n* matches played directly between *home* and *away* (any venue)."""
    mask = (
        ((_df["home_team"] == home) & (_df["away_team"] == away))
        | ((_df["home_team"] == away) & (_df["away_team"] == home))
    )
    h2h = _df[mask].sort_values("date", ascending=False).head(n)
    return h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()


TOURNAMENT_OPTIONS = {
    "Amistoso": "Friendly",
    "Eliminatorias / Clasificacion": "Qualification",
    "Liga de Naciones": "Nations League",
    "Copa continental (Copa America / Eurocopa / etc.)": "Copa America",
    "Mundial": "FIFA World Cup",
    "Juegos Olimpicos": "Olympic Games",
    "Otro": "Other",
}


def render_single_match(teams: list[str]) -> None:
    with st.sidebar:
        st.header("Configuracion del partido")
        home_team = st.selectbox(
            "Equipo local", teams,
            index=teams.index("Argentina") if "Argentina" in teams else 0,
            format_func=lambda t: f"{team_flag(t)} {t}",
        )
        away_team = st.selectbox(
            "Equipo visitante", teams,
            index=teams.index("Brazil") if "Brazil" in teams else 1,
            format_func=lambda t: f"{team_flag(t)} {t}",
        )
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
        return

    if home_team == away_team:
        st.error("Elegi dos selecciones distintas.")
        return

    predictor = get_predictor(weight_bayes, weight_xgb)
    pred = predictor.predict(
        home_team=home_team,
        away_team=away_team,
        is_neutral=is_neutral,
        tournament=tournament,
    )

    home_flag = team_flag(pred.home_team)
    away_flag = team_flag(pred.away_team)

    st.header(f"{home_flag} {pred.home_team}  vs  {pred.away_team} {away_flag}")
    st.caption(f"Torneo: {tournament_label} | Cancha neutral: {is_neutral}")

    # ── Ajuste por clima (si aplica) ──────────────────────────────────────────
    weather_note = None
    poisson_adj = None
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
                weather_note = (match_city, match_date, notes, factor, lh_adj, la_adj)
            else:
                st.caption(f"Clima en {match_city} sin condiciones adversas relevantes — sin ajuste.")

    # ── KPIs principales ───────────────────────────────────────────────────────
    outcomes = [
        (pred.home_win, f"Gana {pred.home_team}", home_flag),
        (pred.draw, "Empate", ""),
        (pred.away_win, f"Gana {pred.away_team}", away_flag),
    ]
    favorite = max(outcomes, key=lambda o: o[0])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(f"{favorite[2]} Resultado mas probable", favorite[1], f"{favorite[0]:.1%}")
    with kpi2:
        st.metric("Marcador mas probable", pred.most_likely_score)
    with kpi3:
        st.metric(f"xG {pred.home_team}", f"{pred.expected_goals_home:.2f}")
    with kpi4:
        st.metric(f"xG {pred.away_team}", f"{pred.expected_goals_away:.2f}")

    summary_png = build_summary_image(pred, tournament_label)
    st.download_button(
        "Descargar resumen (PNG)",
        data=summary_png,
        file_name=f"{pred.home_team}_vs_{pred.away_team}.png".replace(" ", "_"),
        mime="image/png",
    )

    if weather_note is not None:
        match_city_n, match_date_n, notes, factor, lh_adj, la_adj = weather_note
        st.subheader("Ajuste por clima (heuristica, no aprendida por el modelo)")
        st.caption(
            f"{match_city_n} el {match_date_n.date()}: " + ", ".join(notes) +
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

    st.divider()

    tab_resultado, tab_goles, tab_eventos, tab_h2h, tab_modelo = st.tabs(
        ["Resultado", "Goles", "Eventos", "Historial H2H", "Patrones & Modelo"]
    )

    # ── Tab: Resultado ─────────────────────────────────────────────────────────
    with tab_resultado:
        result_df = pd.DataFrame({
            "Resultado": [f"Victoria {pred.home_team}", "Empate", f"Victoria {pred.away_team}"],
            "Probabilidad": [pred.home_win, pred.draw, pred.away_win],
        })
        fig = px.bar(result_df, x="Resultado", y="Probabilidad", text_auto=".1%",
                     color="Resultado", color_discrete_sequence=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR])
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=420,
                           title="Resultado del partido")
        _style_borders(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Goles ──────────────────────────────────────────────────────────────
    with tab_goles:
        col1, col2 = st.columns(2)
        with col1:
            xg_df = pd.DataFrame({
                "Equipo": [pred.home_team, pred.away_team],
                "xG": [pred.expected_goals_home, pred.expected_goals_away],
            })
            fig = px.bar(xg_df, x="Equipo", y="xG", text_auto=".2f", color="Equipo",
                         color_discrete_sequence=[HOME_COLOR, AWAY_COLOR])
            fig.update_layout(showlegend=False, height=380, title="Goles esperados (xG)")
            _style_borders(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            markets_df = pd.DataFrame({
                "Mercado": ["BTTS", "Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5"],
                "Probabilidad": [pred.btts, pred.over_0_5, pred.over_1_5, pred.over_2_5, pred.over_3_5],
            })
            fig = px.bar(markets_df, x="Mercado", y="Probabilidad", text_auto=".1%", color="Probabilidad",
                         color_continuous_scale=ACCENT_SCALE)
            fig.update_layout(yaxis_tickformat=".0%", height=380, coloraxis_showscale=False,
                               title="Mercados de goles")
            _style_borders(fig)
            st.plotly_chart(fig, use_container_width=True)

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
            colorscale=ACCENT_SCALE,
            xgap=3,
            ygap=3,
            texttemplate="%{z:.1%}",
            hovertemplate=f"{pred.home_team}: %{{y}}<br>{pred.away_team}: %{{x}}<br>Prob: %{{z:.1%}}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=f"Goles {pred.away_team}", yaxis_title=f"Goles {pred.home_team}",
            height=420, plot_bgcolor=CREAM_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Eventos ──────────────────────────────────────────────────────────────
    with tab_eventos:
        e = pred.events
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=[pred.home_team, pred.away_team],
                values=[e.home_possession, e.away_possession],
                hole=0.5,
                marker=dict(colors=[HOME_COLOR, AWAY_COLOR], line=dict(color=INK_TEXT, width=BORDER_WIDTH)),
            )])
            fig.update_layout(title="Posesion", height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            shots_df = pd.DataFrame({
                "Equipo": [pred.home_team, pred.away_team, pred.home_team, pred.away_team],
                "Tipo": ["Tiros totales", "Tiros totales", "A puerta", "A puerta"],
                "Valor": [e.home_shots, e.away_shots, e.home_shots_on_target, e.away_shots_on_target],
            })
            fig = px.bar(shots_df, x="Tipo", y="Valor", color="Equipo", barmode="group",
                         color_discrete_sequence=[HOME_COLOR, AWAY_COLOR])
            fig.update_layout(title="Tiros", height=340)
            _style_borders(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            cards_df = pd.DataFrame({
                "Equipo": [pred.home_team, pred.away_team],
                "Amarillas esperadas": [e.home_yellow_cards, e.away_yellow_cards],
            })
            fig = px.bar(cards_df, x="Equipo", y="Amarillas esperadas", text_auto=".2f", color="Equipo",
                         color_discrete_sequence=[CARD_COLOR, CARD_COLOR_2])
            fig.update_layout(title="Tarjetas amarillas", height=340, showlegend=False)
            _style_borders(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Corners esperados: {e.home_corners:.1f} - {e.away_corners:.1f}  |  "
            f"P(roja) {pred.home_team}: {e.home_red_card_prob:.1%}  |  "
            f"P(roja) {pred.away_team}: {e.away_red_card_prob:.1%}"
        )

    # ── Tab: Historial H2H ────────────────────────────────────────────────────────
    with tab_h2h:
        h2h_df = get_h2h_matches(predictor._feature_df, pred.home_team, pred.away_team, n=10)
        if h2h_df.empty:
            st.caption(f"No hay enfrentamientos directos registrados entre {pred.home_team} y {pred.away_team}.")
        else:
            wins_h = int(((h2h_df["home_team"] == pred.home_team) & (h2h_df["home_score"] > h2h_df["away_score"])).sum()
                         + ((h2h_df["away_team"] == pred.home_team) & (h2h_df["away_score"] > h2h_df["home_score"])).sum())
            wins_a = int(((h2h_df["home_team"] == pred.away_team) & (h2h_df["home_score"] > h2h_df["away_score"])).sum()
                         + ((h2h_df["away_team"] == pred.away_team) & (h2h_df["away_score"] > h2h_df["home_score"])).sum())
            draws = len(h2h_df) - wins_h - wins_a

            st.caption(f"Ultimos {len(h2h_df)} enfrentamientos directos (cualquier sede)")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_flag} Victorias {pred.home_team}", wins_h)
            c2.metric("Empates", draws)
            c3.metric(f"{away_flag} Victorias {pred.away_team}", wins_a)

            display_df = h2h_df.copy()
            display_df["Fecha"] = pd.to_datetime(display_df["date"]).dt.strftime("%d/%m/%Y")
            display_df["Resultado"] = (
                display_df["home_team"] + " " + display_df["home_score"].astype(int).astype(str)
                + " - " + display_df["away_score"].astype(int).astype(str) + " " + display_df["away_team"]
            )
            display_df = display_df.rename(columns={"tournament": "Torneo"})
            st.dataframe(
                display_df[["Fecha", "Resultado", "Torneo"]],
                use_container_width=True, hide_index=True,
            )

    # ── Tab: Patrones & Modelo ───────────────────────────────────────────────────
    with tab_modelo:
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
                             text_auto=".1%", color_discrete_map={"FUERTE": HOME_COLOR, "MODERADO": DRAW_COLOR})
                fig.update_layout(xaxis_tickformat=".0%", height=320)
                _style_borders(fig)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No hay suficientes partidos historicos similares para detectar patrones.")

        st.divider()
        st.subheader("Detalle del modelo (Bayesiano vs XGBoost)")
        detail_df = pd.DataFrame({
            "Modelo": ["Bayesiano", "XGBoost", "Ensemble"],
            "xG local": [pred.lambda_bayes_home, pred.lambda_xgb_home, pred.expected_goals_home],
            "xG visitante": [pred.lambda_bayes_away, pred.lambda_xgb_away, pred.expected_goals_away],
            "Peso": [f"{pred.model_weights['bayesian']:.0%}", f"{pred.model_weights['xgboost']:.0%}", "100%"],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        st.caption("Tiros, corners y tarjetas son estimaciones estadisticas basadas en xG, no datos reales.")


def render_quiniela(teams: list[str]) -> None:
    with st.sidebar:
        st.header("Configuracion de la quiniela")
        tournament_label = st.selectbox(
            "Contexto (aplica a todos los partidos)",
            list(TOURNAMENT_OPTIONS.keys()),
        )
        tournament = TOURNAMENT_OPTIONS[tournament_label]
        is_neutral = st.checkbox("Todas a cancha neutral")

        st.divider()
        st.subheader("Pesos del ensemble")
        weight_bayes = st.slider("Peso Bayesiano", 0.0, 1.0, ENSEMBLE_WEIGHT_BAYESIAN, 0.05, key="q_weight")
        weight_xgb = round(1.0 - weight_bayes, 2)
        st.caption(f"Peso XGBoost: {weight_xgb}")

    st.subheader("Armar la fecha")
    st.caption("Agrega filas y elegi local/visitante para cada partido. Hasta 20 partidos por tanda.")

    if "quiniela_rows" not in st.session_state:
        st.session_state.quiniela_rows = pd.DataFrame(
            {"Local": ["Argentina", "Brazil"], "Visitante": ["Brazil", "Argentina"]}
        )

    edited = st.data_editor(
        st.session_state.quiniela_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Local": st.column_config.SelectboxColumn("Local", options=teams, required=True),
            "Visitante": st.column_config.SelectboxColumn("Visitante", options=teams, required=True),
        },
        key="quiniela_editor",
    )
    st.session_state.quiniela_rows = edited

    predict_clicked = st.button("Predecir quiniela", type="primary")

    if not predict_clicked:
        st.info("Completa los partidos arriba y presiona **Predecir quiniela**.")
        return

    valid_rows = [
        (r["Local"], r["Visitante"]) for _, r in edited.iterrows()
        if pd.notna(r["Local"]) and pd.notna(r["Visitante"]) and r["Local"] != r["Visitante"]
    ]
    if not valid_rows:
        st.error("No hay partidos validos (local y visitante deben ser distintos).")
        return

    predictor = get_predictor(weight_bayes, weight_xgb)

    records = []
    for home, away in valid_rows:
        pred = predictor.predict(home_team=home, away_team=away, is_neutral=is_neutral, tournament=tournament)
        outcomes = [
            (pred.home_win, f"{team_flag(home)} {home}"),
            (pred.draw, "Empate"),
            (pred.away_win, f"{team_flag(away)} {away}"),
        ]
        favorite_label, favorite_prob = max(((o[1], o[0]) for o in outcomes), key=lambda x: x[1])
        records.append({
            "Partido": f"{team_flag(home)} {home} vs {away} {team_flag(away)}",
            "1": pred.home_win,
            "X": pred.draw,
            "2": pred.away_win,
            "Favorito": favorite_label,
            "Marcador probable": pred.most_likely_score,
            "xG local": pred.expected_goals_home,
            "xG visitante": pred.expected_goals_away,
            "BTTS": pred.btts,
        })

    st.divider()
    st.subheader(f"Resultados ({len(records)} partidos)")

    results_df = pd.DataFrame(records)
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "1": st.column_config.ProgressColumn("1 (Local)", min_value=0, max_value=1, format="percent"),
            "X": st.column_config.ProgressColumn("X (Empate)", min_value=0, max_value=1, format="percent"),
            "2": st.column_config.ProgressColumn("2 (Visitante)", min_value=0, max_value=1, format="percent"),
            "BTTS": st.column_config.ProgressColumn("BTTS", min_value=0, max_value=1, format="percent"),
            "xG local": st.column_config.NumberColumn("xG local", format="%.2f"),
            "xG visitante": st.column_config.NumberColumn("xG visitante", format="%.2f"),
        },
    )

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar quiniela (CSV)", data=csv, file_name="quiniela.csv", mime="text/csv")


st.title("Predictor de Futbol Internacional")
st.caption("Ensemble Bayesiano (PyMC) + XGBoost Poisson sobre 49k+ partidos historicos (1872-presente)")

mode = st.radio(
    "Modo", ["Partido individual", "Quiniela (varios partidos)"],
    horizontal=True, label_visibility="collapsed",
)
st.divider()

teams = get_team_list()

if mode == "Partido individual":
    render_single_match(teams)
else:
    render_quiniela(teams)
