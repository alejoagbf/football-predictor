"""
CLI prediction script — output completo con eventos de partido.

Uso:
    python predict.py "France" "Iraq"
    python predict.py "Argentina" "Brazil" --neutral --tournament "Copa America"
    python predict.py "Spain" "Germany" --json
    python predict.py "Brazil" "England" --weights 0.7 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import (
    ENSEMBLE_WEIGHT_BAYESIAN,
    ENSEMBLE_WEIGHT_XGBOOST,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
)

logging.basicConfig(
    level=logging.WARNING,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predecir un partido internacional de futbol.")
    p.add_argument("home_team", help="Equipo local")
    p.add_argument("away_team", help="Equipo visitante")
    p.add_argument("--neutral", action="store_true", help="Cancha neutral")
    p.add_argument("--tournament", default="Friendly", help="Nombre del torneo")
    p.add_argument(
        "--weights", nargs=2, type=float,
        metavar=("BAYES", "XGB"),
        default=[ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST],
    )
    p.add_argument("--json", dest="as_json", action="store_true", help="Salida en JSON")
    return p.parse_args()


# ── Helpers de formato ────────────────────────────────────────────────────────

def bar(prob: float, width: int = 22) -> str:
    filled = int(round(prob * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"

def pct(v: float) -> str:
    return f"{v*100:5.1f}%"

def sep(char: str = "-", w: int = 62) -> str:
    return char * w

def header(title: str, w: int = 62) -> str:
    pad = (w - len(title) - 2) // 2
    return "+" + "-" * pad + f" {title} " + "-" * (w - pad - len(title) - 2) + "+"

def row2(label: str, home_val: str, away_val: str, lw: int = 24) -> str:
    return f"  {label:<{lw}} {home_val:>9}   {away_val:>9}"

def market_row(label: str, prob: float, lw: int = 28) -> str:
    return f"  {label:<{lw}} {pct(prob)}  {bar(prob)}"


def fmt_score_table(score_probs: dict[str, float], min_pct: float = 0.5) -> list[str]:
    filtered = {k: v for k, v in score_probs.items() if v * 100 >= min_pct}
    sorted_scores = sorted(filtered.items(), key=lambda x: -x[1])
    lines = [f"  {'Marcador':<10} {'Prob':>7}  {'Barra':}"]
    lines.append("  " + "-" * 52)
    for score, prob in sorted_scores:
        lines.append(f"  {score:<10} {pct(prob)}  {bar(prob, 20)}")
    lines.append("  " + "-" * 52)
    lines.append(f"  Mostrando marcadores con probabilidad >= {min_pct}%")
    return lines


def main() -> None:
    args = parse_args()
    wb, wx = args.weights

    from src.prediction.predictor import MatchPredictor
    predictor = MatchPredictor.load(weight_bayes=wb, weight_xgb=wx)

    pred = predictor.predict(
        home_team=args.home_team,
        away_team=args.away_team,
        is_neutral=args.neutral,
        tournament=args.tournament,
    )

    if args.as_json:
        print(json.dumps(pred.to_dict(), indent=2, default=str))
        return

    e = pred.events
    H = pred.home_team
    A = pred.away_team
    W = 62

    print()
    print("=" * W)
    print(f"  PREDICCION: {H}  vs  {A}")
    print(f"  Torneo: {args.tournament}  |  Neutral: {args.neutral}")
    print("=" * W)

    # ── 1. Resultado del partido ──────────────────────────────────────────────
    print()
    print(header("RESULTADO DEL PARTIDO"))
    print(f"  {'':28} {H:>12}   {A:>12}")
    print("  " + "-" * 56)
    print(f"  {'Victoria local':<28} {pct(pred.home_win):>9}   {bar(pred.home_win)}")
    print(f"  {'Empate':<28} {pct(pred.draw):>9}   {bar(pred.draw)}")
    print(f"  {'Victoria visitante':<28} {pct(pred.away_win):>9}   {bar(pred.away_win)}")

    # ── 2. Goles esperados ────────────────────────────────────────────────────
    print()
    print(header("GOLES ESPERADOS (xG)"))
    print(row2("", H, A))
    print("  " + "-" * 46)
    print(row2("Goles esperados (xG)", f"{pred.expected_goals_home:.2f}", f"{pred.expected_goals_away:.2f}"))
    print(row2("Marcador mas probable", pred.most_likely_score, ""))
    print()
    print(f"  {'Mercado':<30} {'Prob':>7}  {'Barra'}")
    print("  " + "-" * 52)
    print(market_row("Ambos equipos anotan (BTTS)", pred.btts, 30))
    print(market_row("Over 0.5 goles", pred.over_0_5, 30))
    print(market_row("Over 1.5 goles", pred.over_1_5, 30))
    print(market_row("Over 2.5 goles", pred.over_2_5, 30))
    print(market_row("Over 3.5 goles", pred.over_3_5, 30))
    print(market_row(f"Porteria en 0 ({H})", e.home_clean_sheet_prob, 30))
    print(market_row(f"Porteria en 0 ({A})", e.away_clean_sheet_prob, 30))

    # ── 3. Distribucion de marcadores ────────────────────────────────────────
    print()
    print(header("DISTRIBUCION DE MARCADORES"))
    for line in fmt_score_table(pred.score_probabilities, min_pct=0.5):
        print(line)

    # ── 4. Posesion y tiros ───────────────────────────────────────────────────
    print()
    print(header("POSESION Y TIROS"))
    print(row2("", H, A))
    print("  " + "-" * 46)
    print(row2("Posesion (%)", f"{e.home_possession}%", f"{e.away_possession}%"))
    print(row2("Tiros totales (est.)", f"{e.home_shots:.1f}", f"{e.away_shots:.1f}"))
    print(row2("Tiros a puerta (est.)", f"{e.home_shots_on_target:.1f}", f"{e.away_shots_on_target:.1f}"))
    print(row2("Fueras de lugar (est.)", f"{e.home_offsides:.1f}", f"{e.away_offsides:.1f}"))

    # ── 5. Corners ────────────────────────────────────────────────────────────
    print()
    print(header("CORNERS"))
    print(row2("", H, A))
    print("  " + "-" * 46)
    print(row2("Corners esperados", f"{e.home_corners:.1f}", f"{e.away_corners:.1f}"))
    print(row2("Total corners esperados", f"{e.total_corners:.1f}", ""))
    print()
    print(f"  {'Mercado corners':<30} {'Prob':>7}  {'Barra'}")
    print("  " + "-" * 52)
    print(market_row("Over 8.5 corners", e.corners_over_8_5, 30))
    print(market_row("Over 9.5 corners", e.corners_over_9_5, 30))
    print(market_row("Over 10.5 corners", e.corners_over_10_5, 30))
    print(market_row("Over 11.5 corners", e.corners_over_11_5, 30))

    # ── 6. Tarjetas y faltas ──────────────────────────────────────────────────
    print()
    print(header("TARJETAS Y FALTAS"))
    print(row2("", H, A))
    print("  " + "-" * 46)
    print(row2("Tarjetas amarillas (est.)", f"{e.home_yellow_cards:.2f}", f"{e.away_yellow_cards:.2f}"))
    print(row2("Total amarillas esperadas", f"{e.total_yellow_cards:.2f}", ""))
    print(row2("P(tarjeta roja)", f"{e.home_red_card_prob*100:.1f}%", f"{e.away_red_card_prob*100:.1f}%"))
    print(row2("Faltas esperadas (est.)", f"{e.home_fouls:.1f}", f"{e.away_fouls:.1f}"))
    print()
    print(f"  {'Mercado tarjetas':<30} {'Prob':>7}  {'Barra'}")
    print("  " + "-" * 52)
    print(market_row("Over 2.5 tarjetas", e.cards_over_2_5, 30))
    print(market_row("Over 3.5 tarjetas", e.cards_over_3_5, 30))
    print(market_row("Over 4.5 tarjetas", e.cards_over_4_5, 30))
    print(market_row("Over 5.5 tarjetas", e.cards_over_5_5, 30))

    # ── 7. Patrones históricos ────────────────────────────────────────────────
    pat = pred.patterns
    if pat is not None and pat.n_similar_matches >= 10:
        print()
        print(header("PATRONES HISTORICOS SIMILARES"))
        print(f"  Tipo de partido : {pat.elo_bucket_label}")
        print(f"  Partidos similares encontrados: {pat.n_similar_matches}")
        print()

        # Promedios históricos
        avg_h = getattr(pat, "_avg_home_goals", None)
        avg_a = getattr(pat, "_avg_away_goals", None)
        avg_t = getattr(pat, "_avg_total_goals", None)
        if avg_h is not None:
            print(f"  Promedios reales en partidos similares:")
            print(f"    Goles local   : {avg_h:.2f}  (modelo predice {pred.expected_goals_home:.2f})")
            print(f"    Goles visit.  : {avg_a:.2f}  (modelo predice {pred.expected_goals_away:.2f})")
            print(f"    Goles totales : {avg_t:.2f}")
            print()

        # Marcadores más frecuentes históricamente
        top_scores = getattr(pat, "_top_scores", {})
        if top_scores:
            print(f"  Marcadores mas frecuentes historicamente:")
            for score, cnt in list(top_scores.items())[:6]:
                pct_s = cnt / pat.n_similar_matches * 100
                print(f"    {score:<6}  {pct_s:5.1f}%  ({cnt}/{pat.n_similar_matches} partidos)")
            print()

        # Patrones por fuerza
        strong  = [p for p in pat.patterns if p.strength == "FUERTE"]
        moderate = [p for p in pat.patterns if p.strength == "MODERADO"]

        if strong:
            print(f"  PATRONES FUERTES (>= 70% frecuencia historica):")
            print("  " + "-" * 56)
            for p in sorted(strong, key=lambda x: -x.frequency):
                print(f"  {p.description:<42} {pct(p.frequency)}  {bar(p.frequency, 14)}")
            print()

        if moderate:
            print(f"  PATRONES MODERADOS (55-70% frecuencia historica):")
            print("  " + "-" * 56)
            for p in sorted(moderate, key=lambda x: -x.frequency):
                print(f"  {p.description:<42} {pct(p.frequency)}  {bar(p.frequency, 14)}")
            print()

    # ── 8. Detalle del modelo ─────────────────────────────────────────────────
    print()
    print(header("DETALLE DEL MODELO"))
    print(f"  {'Modelo':<12} {'xG local':>10} {'xG visit.':>10}  {'Peso':>6}")
    print("  " + "-" * 44)
    print(f"  {'Bayesiano':<12} {pred.lambda_bayes_home:>10.3f} {pred.lambda_bayes_away:>10.3f}  {pred.model_weights['bayesian']:>5.0%}")
    print(f"  {'XGBoost':<12} {pred.lambda_xgb_home:>10.3f} {pred.lambda_xgb_away:>10.3f}  {pred.model_weights['xgboost']:>5.0%}")
    print(f"  {'Ensemble':<12} {pred.expected_goals_home:>10.3f} {pred.expected_goals_away:>10.3f}  {'100%':>6}")
    print()
    print("  NOTA: Tiros, corners y tarjetas son estimaciones estadisticas")
    print("  basadas en xG y promedios del futbol internacional.")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
