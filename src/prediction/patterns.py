"""
Empirical pattern analyzer.

Busca en los 49k partidos históricos los casos similares al partido
predicho (por diferencia de ELO, tipo de torneo, ventaja de localía)
y calcula frecuencias empíricas reales para validar / complementar
las predicciones del modelo.

Los patrones se clasifican por "fuerza":
  FUERTE  >= 70% de frecuencia histórica
  MODERADO 55-70%
  DEBIL   < 55%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Umbrales para clasificar diferencias de ELO ──────────────────────────────
ELO_BINS = [
    (400,  9999, "Dominador absoluto  (ELO diff > 400)"),
    (250,   400, "Favorito claro      (ELO diff 250-400)"),
    (100,   250, "Ligero favorito     (ELO diff 100-250)"),
    (  0,   100, "Partido equilibrado (ELO diff 0-100)"),
    (-100,    0, "Ligero visitante    (ELO diff 0-100, visitante mejor)"),
    (-250, -100, "Visitante favorito  (ELO diff 100-250)"),
    (-9999,-250, "Visitante dominante (ELO diff > 250)"),
]


@dataclass
class Pattern:
    """Un patrón estadístico encontrado en partidos históricos similares."""
    description: str
    frequency: float      # 0-1
    sample_size: int
    strength: str         # FUERTE / MODERADO / DEBIL
    category: str         # goles / resultado / tarjetas / corners


@dataclass
class PatternReport:
    """Conjunto de patrones encontrados para un partido específico."""
    elo_diff: float
    elo_bucket_label: str
    n_similar_matches: int
    patterns: list[Pattern]

    def strong(self) -> list[Pattern]:
        return [p for p in self.patterns if p.strength == "FUERTE"]

    def moderate(self) -> list[Pattern]:
        return [p for p in self.patterns if p.strength == "MODERADO"]


def _elo_label(elo_diff: float) -> str:
    for lo, hi, label in ELO_BINS:
        if lo <= elo_diff < hi:
            return label
    return "Desconocido"


def _strength(freq: float) -> str:
    if freq >= 0.70:
        return "FUERTE"
    if freq >= 0.55:
        return "MODERADO"
    return "DEBIL"


def _pattern(desc: str, freq: float, n: int, category: str) -> Pattern:
    return Pattern(
        description=desc,
        frequency=freq,
        sample_size=n,
        strength=_strength(freq),
        category=category,
    )


def analyze_patterns(
    df: pd.DataFrame,
    elo_diff: float,
    lambda_home: float,
    lambda_away: float,
    is_neutral: bool,
    tournament_category: str = "friendly",
    home_team: str = "",
    away_team: str = "",
    elo_tolerance: float = 80.0,
    min_sample: int = 30,
) -> PatternReport:
    """
    Encuentra partidos históricos similares y calcula patrones empíricos.

    Similitud = diferencia de ELO dentro de ±elo_tolerance puntos.
    Si hay pocos partidos similares, amplía el rango automáticamente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con features completos (salida de FeaturePipeline).
    elo_diff : float
        Diferencia ELO del partido a predecir (home_elo - away_elo).
    lambda_home, lambda_away : float
        xG esperado de cada equipo (para filtrar por nivel ofensivo).
    is_neutral : bool
        Venue neutral o no.
    tournament_category : str
        Categoría del torneo (friendly, world_cup, etc.).
    elo_tolerance : float
        Tolerancia inicial en puntos ELO para buscar partidos similares.
    min_sample : int
        Mínimo de partidos para calcular estadísticas confiables.
    """
    # Asegurar columnas necesarias
    needed = {"elo_diff", "home_score", "away_score", "neutral"}
    if not needed.issubset(df.columns):
        return PatternReport(elo_diff=elo_diff, elo_bucket_label="Sin datos",
                             n_similar_matches=0, patterns=[])

    # Ampliar tolerancia si hay pocos partidos
    tol = elo_tolerance
    for _ in range(6):
        mask = (df["elo_diff"] >= elo_diff - tol) & (df["elo_diff"] < elo_diff + tol)
        if is_neutral:
            mask &= df["neutral"].astype(bool)
        subset = df[mask].copy()
        if len(subset) >= min_sample:
            break
        tol *= 1.5
    else:
        # Sin tolerancia de neutral si sigue siendo poco
        mask = (df["elo_diff"] >= elo_diff - tol) & (df["elo_diff"] < elo_diff + tol)
        subset = df[mask].copy()

    n = len(subset)
    if n < 10:
        return PatternReport(
            elo_diff=elo_diff,
            elo_bucket_label=_elo_label(elo_diff),
            n_similar_matches=n,
            patterns=[],
        )

    # ── Variables derivadas ────────────────────────────────────────────────────
    subset = subset.copy()
    subset["total_goals"]  = subset["home_score"] + subset["away_score"]
    subset["home_win"]     = (subset["home_score"] > subset["away_score"]).astype(int)
    subset["draw"]         = (subset["home_score"] == subset["away_score"]).astype(int)
    subset["away_win"]     = (subset["home_score"] < subset["away_score"]).astype(int)
    subset["btts"]         = ((subset["home_score"] >= 1) & (subset["away_score"] >= 1)).astype(int)
    subset["over_05"]      = (subset["total_goals"] >= 1).astype(int)
    subset["over_15"]      = (subset["total_goals"] >= 2).astype(int)
    subset["over_25"]      = (subset["total_goals"] >= 3).astype(int)
    subset["over_35"]      = (subset["total_goals"] >= 4).astype(int)
    subset["home_cs"]      = (subset["away_score"] == 0).astype(int)
    subset["away_cs"]      = (subset["home_score"] == 0).astype(int)
    subset["home_2plus"]   = (subset["home_score"] >= 2).astype(int)
    subset["home_3plus"]   = (subset["home_score"] >= 3).astype(int)
    subset["away_score_1plus"] = (subset["away_score"] >= 1).astype(int)

    patterns: list[Pattern] = []

    def add(desc: str, col: str, cat: str) -> None:
        freq = float(subset[col].mean())
        patterns.append(_pattern(desc, freq, n, cat))

    # ── Resultado ─────────────────────────────────────────────────────────────
    add("Victoria del local",               "home_win",  "resultado")
    add("Empate",                            "draw",      "resultado")
    add("Victoria del visitante",            "away_win",  "resultado")

    # ── Goles ─────────────────────────────────────────────────────────────────
    add("Ambos equipos anotan (BTTS)",       "btts",      "goles")
    add("Over 0.5 goles",                    "over_05",   "goles")
    add("Over 1.5 goles",                    "over_15",   "goles")
    add("Over 2.5 goles",                    "over_25",   "goles")
    add("Over 3.5 goles",                    "over_35",   "goles")
    add("Porteria a cero del local",         "home_cs",   "goles")
    add("Porteria a cero del visitante",     "away_cs",   "goles")
    add("Local anota 2 o mas goles",         "home_2plus","goles")
    add("Local anota 3 o mas goles",         "home_3plus","goles")
    add("Visitante anota al menos 1 gol",    "away_score_1plus", "goles")

    # ── Promedios de goles ────────────────────────────────────────────────────
    avg_home  = float(subset["home_score"].mean())
    avg_away  = float(subset["away_score"].mean())
    avg_total = float(subset["total_goals"].mean())

    # Marcadores más frecuentes
    score_counts = (
        subset.assign(score=subset["home_score"].astype(str) + "-" + subset["away_score"].astype(str))
        ["score"].value_counts()
    )

    return PatternReport(
        elo_diff=elo_diff,
        elo_bucket_label=_elo_label(elo_diff),
        n_similar_matches=n,
        patterns=patterns,
        # Extras accesibles desde fuera
        # (los guardamos como atributos adicionales en el objeto)
    ) | _extras(avg_home, avg_away, avg_total, score_counts, n, patterns)


def _extras(avg_home, avg_away, avg_total, score_counts, n, patterns) -> dict:
    """Devuelve atributos extra que se inyectan en PatternReport via __or__."""
    return {
        "_avg_home_goals":  avg_home,
        "_avg_away_goals":  avg_away,
        "_avg_total_goals": avg_total,
        "_top_scores":      score_counts.head(8).to_dict(),
        "_n":               n,
    }


# Monkey-patch para soportar | dict en dataclass
_orig_init = PatternReport.__init__

def _new_or(self, other: dict) -> "PatternReport":
    for k, v in other.items():
        object.__setattr__(self, k, v)
    return self

PatternReport.__or__ = _new_or  # type: ignore[attr-defined]
