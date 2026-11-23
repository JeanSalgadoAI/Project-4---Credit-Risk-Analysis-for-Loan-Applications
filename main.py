"""Análises reproduzíveis sobre o CSV sintético versionado neste repositório.

Execute: python main.py. Não instala pacotes, não baixa nem sobrescreve dados.
"""
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42


def prepare(df, cfg):
    """Seleciona apenas atributos definidos antes da avaliação."""
    frame = df.copy()
    if cfg.get("derive_target"):
        spec = cfg["derive_target"]
        frame[spec["name"]] = (frame[spec["source"]] >= spec["minimum"]).astype(int)
    features = list(cfg.get("features", []))
    if cfg.get("date"):
        dt = pd.to_datetime(frame[cfg["date"]], errors="raise")
        frame["calendar_month"] = dt.dt.month
        frame["calendar_weekday"] = dt.dt.dayofweek
        frame["calendar_hour"] = dt.dt.hour
        features += ["calendar_month", "calendar_weekday"]
        if cfg.get("hour"):
            features += ["calendar_hour"]
    if cfg.get("target") in features:
        raise ValueError("A variável alvo não pode ser usada como atributo.")
    for forbidden in cfg.get("forbidden_features", []):
        if forbidden in features:
            raise ValueError(f"Atributo indisponível na previsão: {forbidden}")
    return frame, features


def split_indices(frame, cfg):
    """Separa por tempo, grupo ou amostra estratificada conforme o problema."""
    idx = np.arange(len(frame))
    if cfg.get("split") == "time":
        dates = pd.to_datetime(frame[cfg["date"]])
        unique_dates = np.sort(dates.unique())
        cutoff = unique_dates[max(1, int(len(unique_dates) * .8))]
        train = idx[dates < cutoff]
        test = idx[dates >= cutoff]
        assert dates.iloc[train].max() < dates.iloc[test].min()
    elif cfg.get("split") == "group":
        groups = frame[cfg["group"]]
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=SEED).split(idx, groups=groups))
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    else:
        stratify = frame[cfg["target"]] if cfg["mode"] in ("classification", "text") else None
        train, test = train_test_split(idx, test_size=.3, random_state=SEED, stratify=stratify)
    assert set(train).isdisjoint(test) and len(train) + len(test) == len(frame)
    if not len(train) or not len(test):
        raise ValueError("Separação sem amostras de treino ou teste.")
    return train, test


def preprocessor(frame, features, categorical):
    cats = [c for c in features if c in categorical or not pd.api.types.is_numeric_dtype(frame[c])]
    nums = [c for c in features if c not in cats]
    transformations = []
    if nums:
        transformations.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), nums))
    if cats:
        transformations.append(("category", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    return ColumnTransformer(transformations, remainder="drop")


def classify_metrics(y, pred, prob):
    result = {"accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y)) == 2:
        result.update(roc_auc=float(roc_auc_score(y, prob)),
                      average_precision=float(average_precision_score(y, prob)))
    else:
        result.update(roc_auc=None, average_precision=None)
    return result


def regress_metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(math.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def evaluate(frame, cfg, features):
    train, test = split_indices(frame, cfg)
    y = frame[cfg["target"]]
    classifier = cfg["mode"] in ("classification", "text")
    if classifier and set(y.unique()) != {0, 1}:
        raise ValueError("A classificação deste estudo requer alvo binário 0/1.")
    if cfg["mode"] == "text":
        x = frame[cfg["text"]].fillna("")
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                         ("model", LogisticRegression(max_iter=1000, random_state=SEED))])
    else:
        x = frame[features]
        model = (RandomForestClassifier(n_estimators=120, min_samples_leaf=3, random_state=SEED, n_jobs=1)
                 if classifier else
                 LinearRegression() if cfg.get("linear") else
                 RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=SEED, n_jobs=1))
        pipe = Pipeline([("prepare", preprocessor(frame, features, cfg.get("categorical", []))),
                         ("model", model)])
    # Todos os parâmetros aprendidos, incluindo vocabulário e escala, usam só treino.
    pipe.fit(x.iloc[train], y.iloc[train])
    pred = pipe.predict(x.iloc[test])
    dummy_x = np.zeros((len(frame), 1))
    baseline = DummyClassifier(strategy="prior") if classifier else DummyRegressor(strategy="mean")
    baseline.fit(dummy_x[train], y.iloc[train])
    base_pred = baseline.predict(dummy_x[test])
    if classifier:
        prob = pipe.predict_proba(x.iloc[test])[:, list(pipe.classes_).index(1)]
        base_prob = baseline.predict_proba(dummy_x[test])[:, list(baseline.classes_).index(1)]
        model_metrics = classify_metrics(y.iloc[test], pred, prob)
        baseline_metrics = classify_metrics(y.iloc[test], base_pred, base_prob)
        compare = ["balanced_accuracy", "f1", "average_precision", "roc_auc"]
    else:
        model_metrics = regress_metrics(y.iloc[test], pred)
        baseline_metrics = regress_metrics(y.iloc[test], base_pred)
        compare = ["mae", "rmse"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(compare))
    ax.bar(positions - .18, [model_metrics.get(k) or 0 for k in compare], .36,
           label="Modelo", color="#167D9A")
    ax.bar(positions + .18, [baseline_metrics.get(k) or 0 for k in compare], .36,
           label="Referência simples", color="#ADB9C6")
    ax.set_xticks(positions, [k.replace("_", " ") for k in compare])
    ax.set_ylabel("Pontuação (maior é melhor)" if classifier else "Erro (menor é melhor)")
    ax.set_title("Avaliação em amostra reservada • dados sintéticos")
    ax.legend(frameon=False)
    fig.tight_layout()
    detail = pd.DataFrame({"row_index": test, "actual": y.iloc[test].to_numpy(),
                           "prediction": pred, "baseline_prediction": base_pred})
    if classifier:
        detail["probability_1"] = prob
    result = {"task": "classification" if classifier else "regression",
        "split": cfg.get("split", "stratified" if classifier else "random"),
        "train_rows": len(train), "test_rows": len(test), "features": features or [cfg.get("text")],
        "target": cfg["target"], "model": type(pipe.named_steps["model"]).__name__,
        "baseline": type(baseline).__name__, "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics}
    if classifier:
        result["test_positive_rate"] = float(y.iloc[test].mean())
    if cfg.get("split") == "time":
        result["train_end"] = str(frame.iloc[train][cfg["date"]].max())
        result["test_start"] = str(frame.iloc[test][cfg["date"]].min())
    if cfg.get("split") == "group":
        result["train_groups"] = sorted(frame.iloc[train][cfg["group"]].unique().tolist())
        result["test_groups"] = sorted(frame.iloc[test][cfg["group"]].unique().tolist())
    return result, detail, fig


def rating_quality(df, cfg):
    counts = pd.crosstab(df[cfg["text"]], df[cfg["rating"]]).reindex(columns=range(1, 6), fill_value=0)
    sizes = counts.sum(axis=1)
    conflicts = (counts > 0).sum(axis=1) > 1
    detail = counts.copy()
    detail["rows"] = sizes
    detail["distinct_ratings"] = (counts > 0).sum(axis=1)
    detail["dominant_rating_share"] = counts.max(axis=1) / sizes
    detail = detail.reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    totals = counts.sum(axis=0)
    ax.bar(totals.index, totals.values, color="#167D9A")
    ax.set(xlabel="Nota simulada", ylabel="Registros", title="Distribuição das notas • dados sintéticos")
    ax.set_xticks(range(1, 6))
    fig.tight_layout()
    return {"task": "data_quality", "unique_texts": len(counts),
        "texts_with_multiple_ratings": int(conflicts.sum()),
        "fraction_rows_in_conflicting_texts": float(sizes[conflicts].sum() / len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "model_trained": False,
        "reason": "Notas independentes dos textos e poucos textos distintos não validam um classificador de sentimento."}, detail, fig


def forecast_stores(df, cfg):
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    dates = np.sort(frame.Date.unique())
    cutoff = dates[int(len(dates)*.8)]
    train, test = frame[frame.Date < cutoff], frame[frame.Date >= cutoff]
    assert train.Date.max() < test.Date.min()
    chunks = []
    for store, history in train.groupby("Store"):
        history = history.sort_values("Date")
        future = test[test.Store == store].sort_values("Date").copy()
        future["prediction_last"] = history.Sales.iloc[-1]
        future["prediction_mean4"] = history.Sales.tail(4).mean()
        future["prediction_mean"] = history.Sales.mean()
        chunks.append(future)
    detail = pd.concat(chunks, ignore_index=True)
    metrics = {label: regress_metrics(detail.Sales, detail[col]) for label, col in
               [("last_value", "prediction_last"), ("mean_last_4", "prediction_mean4"), ("train_mean", "prediction_mean")]}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(list(metrics), [m["mae"] for m in metrics.values()], color=["#167D9A", "#66AFBD", "#ADB9C6"])
    ax.set(ylabel="MAE (menor é melhor)", title="Previsão com origem fixa • cinco lojas simuladas")
    fig.tight_layout()
    return {"task": "time_series_baselines", "train_rows": len(train), "test_rows": len(test),
        "train_end": str(train.Date.max()), "test_start": str(test.Date.min()),
        "stores": int(frame.Store.nunique()), "metrics": metrics,
        "protocol": "Origem fixa: nenhum valor observado no teste atualiza as previsões."}, detail, fig


def ppc_summary(df, cfg):
    revenue_per_conversion = 20.0
    with sqlite3.connect(":memory:") as connection:
        df.to_sql("campaigns", connection, index=False)
        query = (ROOT / "queries/campaign_summary.sql").read_text(encoding="utf-8")
        detail = pd.read_sql_query(query, connection)
    expected = df.Clicks * df.CPC
    if not np.allclose(df.Cost, expected):
        raise ValueError("Custo inconsistente com Clicks * CPC.")
    totals = {c: float(df[c].sum()) for c in ("Impressions", "Clicks", "Conversions", "Cost")}
    def ratio(a, b):
        return float(a / b) if b else None
    result = {"task": "campaign_reporting", "campaigns": len(detail),
        "assumed_revenue_per_conversion": revenue_per_conversion,
        "totals": totals,
        "ctr_percent": ratio(100*totals["Clicks"], totals["Impressions"]),
        "conversion_rate_percent": ratio(100*totals["Conversions"], totals["Clicks"]),
        "roas": ratio(totals["Conversions"]*revenue_per_conversion, totals["Cost"]),
        "modeled_return_on_ad_spend": ratio(totals["Conversions"]*revenue_per_conversion-totals["Cost"], totals["Cost"])}
    chosen = detail.sort_values("cost", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(chosen.Campaign_ID.astype(str), chosen.cost, color="#167D9A")
    ax.set(xlabel="Identificador sintético", ylabel="Custo (unidade monetária)", title="Dez campanhas com maior gasto simulado")
    fig.tight_layout()
    return result, detail, fig


def render_report(cfg, results, output):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output / "report.pdf", metadata={"Title": cfg["title"], "Author": "", "Subject": "Estudo com dados sintéticos"}) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.09, .94, f"PROJETO {cfg['number']:02d}  |  ANÁLISE REPRODUZÍVEL", fontsize=10, color="#167D9A")
        title_lines = textwrap.wrap(cfg["title"], 47)
        fig.text(.09, .89, "\n".join(title_lines), fontsize=19, weight="bold", va="top")
        y = .80
        paragraphs = [cfg["objective"], "Dados: base sintética versionada; não representa clientes ou operações reais.",
                      f"Amostra: {results['data_rows']} linhas. Arquivo: {cfg['data']}",
                      "Limites: " + cfg["limitations"],
                      "Reprodução: python main.py. Métricas completas, previsões e hash do CSV em reports/."]
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, 88)
            fig.text(.09, y, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.5)
            y -= .021*len(lines) + .025
        ax = fig.add_axes([.14, .16, .76, .22])
        ax.set_title("Resultados da execução", fontsize=11, loc="left", pad=12)
        if results["task"] in ("classification", "regression"):
            keys = ["balanced_accuracy", "f1", "average_precision", "roc_auc"] if results["task"] == "classification" else ["mae", "rmse"]
            ticks = np.arange(len(keys))
            a = ax.bar(ticks-.18, [results["model_metrics"][k] or 0 for k in keys], .36, color="#167D9A", label="Modelo")
            b = ax.bar(ticks+.18, [results["baseline_metrics"][k] or 0 for k in keys], .36, color="#ADB9C6", label="Referência")
            ax.set_xticks(ticks, [k.replace("balanced_accuracy", "Bal. accuracy").replace("average_precision", "Avg. precision").replace("roc_auc", "ROC AUC").upper() for k in keys])
            ax.bar_label(a, fmt="%.3f", fontsize=8, padding=3)
            ax.bar_label(b, fmt="%.3f", fontsize=8, padding=3)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax.margins(y=.3)
        elif results["task"] == "data_quality":
            bars=ax.bar(["Textos distintos", "Textos com notas diferentes"], [results["unique_texts"], results["texts_with_multiple_ratings"]], color=["#167D9A", "#ADB9C6"])
            ax.bar_label(bars, padding=3, fontsize=9)
            ax.margins(y=.2)
        elif results["task"] == "time_series_baselines":
            bars=ax.bar(["Último valor", "Média de 4 semanas", "Média histórica"], [v["mae"] for v in results["metrics"].values()], color="#167D9A")
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            ax.set_ylabel("MAE")
            ax.margins(y=.2)
        else:
            bars=ax.bar(["CTR (%)", "Conversão (%)"], [results["ctr_percent"], results["conversion_rate_percent"]], color="#167D9A")
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            ax.margins(y=.2)
            ax.text(0, -.3, f"ROAS simulado: {results['roas']:.3f} | Receita assumida por conversão: 20", transform=ax.transAxes, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(.09, .075, "Resultados são demonstrações metodológicas; não sustentam impacto comercial real.", fontsize=9, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)


def run(root=None):
    global ROOT
    if root is not None:
        ROOT = Path(root).resolve()
    cfg = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    data_path = ROOT / cfg["data"]
    df = pd.read_csv(data_path)
    if df.empty or not df.columns.is_unique:
        raise ValueError("CSV vazio ou com colunas duplicadas.")
    frame, features = prepare(df, cfg)
    if cfg["mode"] == "ratings":
        results, detail, fig = rating_quality(df, cfg)
    elif cfg["mode"] == "stores":
        results, detail, fig = forecast_stores(df, cfg)
    elif cfg["mode"] == "ppc":
        results, detail, fig = ppc_summary(df, cfg)
    else:
        results, detail, fig = evaluate(frame, cfg, features)
    results.update(data_rows=len(df), data_columns=len(df.columns),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        missing_cells=int(df.isna().sum().sum()), random_seed=SEED,
        data_origin="synthetic", limitations=cfg["limitations"])
    if cfg["number"] == 16:
        results["rows_click_rate_above_open_rate"] = int((df.click_rate > df.open_rate).sum())
        results["rate_caveat"] = "Taxas geradas separadamente: sem denominadores, não constituem um funil validado."
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "summary.svg", metadata={"Date": None, "Creator": "Matplotlib"})
    plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    detail.to_csv(output / "results.csv", index=False)
    report = f"# {cfg['title']}\n\n{cfg['objective']}\n\n## Limitações\n\n{cfg['limitations']}\n\n## Resultados calculados\n\n```json\n{json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)}\n```\n\n![Resumo](summary.svg)\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    render_report(cfg, results, output)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
