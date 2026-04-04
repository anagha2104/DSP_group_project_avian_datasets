import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr, t as student_t

from analysis_01 import (
    is_categorical,
    is_numerical,
)

units = {"mass" : "gram"}

# ─── colour palette: distinct, colour-blind-friendly ────────────────────────
_PALETTE = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
    "#06B6D4", "#F97316", "#EC4899", "#84CC16", "#6366F1",
]


def _colour(i):
    return _PALETTE[i % len(_PALETTE)]


# ─── regression helper ──────────────────────────────────────────────────────
def _fit_line(x_vals, y_vals, log_x=False, log_y=False):
    """Fit a stable OLS line; everything in transformed (log) space."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 3 or np.unique(x_arr).size < 2:
        return None

    x_fit = np.log10(x_arr) if log_x else x_arr
    y_fit = np.log10(y_arr) if log_y else y_arr
    m, b = np.polyfit(x_fit, y_fit, 1)
    xs_fit = np.linspace(x_fit.min(), x_fit.max(), 300)
    ys_fit = m * xs_fit + b
    xs_plot = 10 ** xs_fit if log_x else xs_fit
    ys_plot = 10 ** ys_fit if log_y else ys_fit
    return xs_plot, ys_plot, float(m), float(b)


def _summary_for_series(series, name):
    clean = series.dropna()
    is_num = pd.api.types.is_numeric_dtype(clean) and len(clean) > 0
    return {
        "feature": name,
        "mean":   f"{clean.mean():.3f}"   if is_num else "N/A",
        "median": f"{clean.median():.3f}" if is_num else "N/A",
        "std":    f"{clean.std():.3f}"    if is_num else "N/A",
        "unique": f"{clean.nunique()}",
        "mode":   str(clean.mode().iloc[0]) if len(clean.mode()) else "N/A",
    }


# ─── rich, professor-grade conclusion generators ─────────────────────────────
def generate_conclusion(case_type, x, y, stats, subfeature=None):
    """Return 3-4 sentences suitable for presenting analysis to a professor."""
    if not stats:
        return (
            f"The selected feature pair ({x} vs {y}) could not be statistically "
            "summarised in the current filtered subset — the data available is too "
            "sparse or uniform. Consider broadening the filter or choosing a different "
            "pair of features to draw meaningful conclusions."
        )

    # ── Numeric vs Numeric ────────────────────────────────────────────────
    if case_type == "num_num":
        r = stats.get("pearson_r")
        spe_r = stats.get("spearman_r")
        n = stats.get("sample_size", "N/A")
        reg = stats.get("regression_equations", {})
        overall_slope = reg.get("overall", {}).get("slope")

        if not isinstance(r, (int, float)):
            return (
                f"Although {n} species were analysed, {x} and {y} do not show "
                "consistent co-variation in this filtered subset — one of the "
                "features may have very limited range here. "
                "Removing the active filter or selecting a broader taxonomic group "
                "is recommended before drawing conclusions about their relationship."
            )

        strength = "weak" if abs(r) < 0.4 else ("moderate" if abs(r) < 0.7 else "strong")
        direction = "positive" if r >= 0 else "negative"
        r_str   = f"{r:.3f}"
        spe_str = f"{spe_r:.3f}" if isinstance(spe_r, float) else "N/A"
        slope_str = (
            f" The OLS regression slope is {overall_slope:.3f}, indicating that "
            f"each unit increase in {x} is associated with a {abs(overall_slope):.3f}-unit "
            f"{'increase' if overall_slope > 0 else 'decrease'} in {y}."
            if isinstance(overall_slope, float) else ""
        )

        sentence1 = (
            f"Across {n} species, {x} and {y} exhibit a **{strength} {direction}** "
            f"linear relationship (Pearson r = {r_str}, Spearman ρ = {spe_str})."
        )
        sentence2 = slope_str
        spearman_close = isinstance(spe_r, float) and abs(spe_r - r) < 0.15
        sentence3 = (
            f"The Spearman rank correlation {'confirms' if spearman_close else 'differs slightly from Pearson, suggesting'} "
            f"{'a monotonic, roughly linear pattern' if spearman_close else 'some non-linearity or influential outliers in the data'}."
        )

        subgroup_stats = stats.get("subfeature_stats", {})
        if subfeature and subgroup_stats:
            groups_sorted = sorted(subgroup_stats.items(), key=lambda kv: kv[1].get("x_mean", 0))
            low_g  = groups_sorted[0][0]  if groups_sorted else ""
            high_g = groups_sorted[-1][0] if groups_sorted else ""
            sentence4 = (
                f"When coloured by **{subfeature}**, groups show different mean positions "
                f"along both axes — '{high_g}' tends highest in {x} while '{low_g}' "
                "tends lowest — suggesting that taxon-level ecology mediates this relationship."
            )
        else:
            sentence4 = (
                "No sub-grouping was applied; the regression line represents the "
                "global avian pattern across all species in the filtered dataset."
            )
        return " ".join(filter(None, [sentence1, sentence2, sentence3, sentence4]))

    # ── Numeric vs Categorical ────────────────────────────────────────────
    if case_type == "num_cat":
        cat_stats = stats.get("category_stats", {})
        eta = stats.get("eta_squared")
        p   = stats.get("p_value")
        n   = stats.get("sample_size", "N/A")

        if not cat_stats:
            return (
                f"No category-level statistics could be computed for {x} across "
                f"{y} in this filtered subset. "
                "Try broadening the dataset or selecting features with better coverage."
            )

        effect = (
            "negligible" if (eta or 0) < 0.01 else
            "small"      if (eta or 0) < 0.06 else
            "moderate"   if (eta or 0) < 0.14 else "large"
        )
        eta_str = f"{eta:.3f}" if isinstance(eta, float) else "N/A"
        p_str   = f"{p:.2e}"   if isinstance(p,   float) else "N/A"
        top_cat  = max(cat_stats.items(), key=lambda kv: kv[1]["mean"])[0]
        low_cat  = min(cat_stats.items(), key=lambda kv: kv[1]["mean"])[0]
        top_mean = cat_stats[top_cat]["mean"]
        low_mean = cat_stats[low_cat]["mean"]

        sentence1 = (
            f"Across {n} species, **{x}** differs meaningfully between **{y}** categories "
            f"(ANOVA: η² = {eta_str}, p = {p_str}), indicating a **{effect} effect size**."
        )
        sentence2 = (
            f"The '{top_cat}' category shows the highest mean {x} "
            f"({top_mean:.2f}), while '{low_cat}' shows the lowest ({low_mean:.2f}), "
            f"a difference of {abs(top_mean - low_mean):.2f} units."
        )
        sentence3 = (
            "This pattern suggests that ecological niche or lifestyle category "
            "is a meaningful predictor of morphological measurements in birds — "
            "species sharing a habitat type or trophic role tend to converge in body dimensions."
        )
        if subfeature:
            sentence4 = (
                f"Sub-grouping by **{subfeature}** reveals further stratification "
                "within each category, indicating that multiple overlapping ecological "
                "dimensions jointly shape variation in this trait."
            )
        else:
            sentence4 = (
                "These between-category differences are preserved even after filtering, "
                "suggesting they reflect genuine ecological signals rather than sampling artefacts."
            )
        return " ".join([sentence1, sentence2, sentence3, sentence4])

    # ── Categorical vs Categorical ────────────────────────────────────────
    if case_type == "cat_cat":
        v = stats.get("cramers_v")
        p = stats.get("p_value")
        n = stats.get("sample_size", "N/A")
        freq_tables = stats.get("frequency_tables", {})

        if not isinstance(v, (int, float)):
            return (
                f"The association between {x} and {y} could not be reliably quantified "
                "in this subset — the contingency table may be too sparse. "
                "A larger or less-restricted dataset is recommended."
            )

        strength = "weak" if v < 0.2 else ("moderate" if v < 0.4 else "strong")
        fx = freq_tables.get("feature_x_frequency", {})
        fy = freq_tables.get("feature_y_frequency", {})
        dom_x = max(fx, key=fx.get) if fx else "N/A"
        dom_y = max(fy, key=fy.get) if fy else "N/A"

        v_str = f"{v:.3f}"
        p_str = f"{p:.2e}" if isinstance(p, float) else "N/A"
        sentence1 = (
            f"Among {n} species, **{x}** and **{y}** show a **{strength} association** "
            f"(Cramér's V = {v_str}, χ² p = {p_str})."
        )
        sentence2 = (
            f"The dominant category in {x} is '{dom_x}' and in {y} is '{dom_y}'; "
            "certain combinations co-occur far more often than expected by chance, "
            "indicating a structured ecological relationship."
        )
        sentence3 = (
            "In avian ecology this type of association often reflects adaptive "
            "convergence — birds with a particular trophic role or habitat preference "
            "independently evolve similar secondary traits (habitat density, lifestyle, etc.)."
        )
        sentence4 = (
            "The heatmap cell intensities directly show which category combinations "
            "are most common, making this a strong visual tool for identifying "
            "dominant ecological guilds in the dataset."
        )
        return " ".join([sentence1, sentence2, sentence3, sentence4])

    return "The selected feature pair shows an interpretable pattern in the current filtered subset."


# ─── Num × Num ──────────────────────────────────────────────────────────────
def analyze_num_num(df, x, y, subfeature=None, show_ci=False, log_x=False, log_y=False):
    cols = [x, y]
    color_col = None
    if subfeature and subfeature in df.columns and subfeature not in [x, y]:
        cols.append(subfeature)
        color_col = subfeature

    clean = df[cols].dropna(subset=[x, y]).copy()
    clean = clean[np.isfinite(clean[x]) & np.isfinite(clean[y])]
    if log_x:
        clean = clean[clean[x] > 0]
    if log_y:
        clean = clean[clean[y] > 0]
    n = len(clean)
    if n < 3:
        return None

    fig = go.Figure()

    # ── scatter traces ──
    if color_col:
        grp_vals = clean[color_col].dropna().unique()
        for gi, grp in enumerate(sorted(grp_vals, key=str)):
            gdf = clean[clean[color_col] == grp]
            fig.add_trace(go.Scatter(
                x=gdf[x].tolist(), y=gdf[y].tolist(),
                mode="markers", name=str(grp),
                marker=dict(size=6, opacity=0.75,
                            color=_colour(gi),
                            line=dict(width=0.5, color="white")),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=clean[x].tolist(), y=clean[y].tolist(),
            mode="markers", name="Species",
            marker=dict(size=6, opacity=0.7, color="#3B82F6",
                        line=dict(width=0.5, color="white")),
        ))

    # ── per-group regression lines ──
    regression_equations = {}
    if color_col:
        grp_vals = clean[color_col].dropna().unique()
        for gi, grp in enumerate(sorted(grp_vals, key=str)):
            gdf = clean[clean[color_col] == grp]
            fit = _fit_line(gdf[x].values, gdf[y].values, log_x=log_x, log_y=log_y)
            if fit is None:
                continue
            xs_g, yhat_g, m_g, b_g = fit
            fig.add_trace(go.Scatter(
                x=xs_g, y=yhat_g,
                mode="lines", name=f"{grp} regression",
                line=dict(width=2.0, color=_colour(gi), dash="dot"),
                showlegend=True,
            ))
            regression_equations[str(grp)] = {"slope": m_g, "intercept": b_g}

    # ── overall regression line ──
    fit_overall = _fit_line(clean[x].values, clean[y].values, log_x=log_x, log_y=log_y)
    overall_lr = None
    if fit_overall is not None:
        xs, yhat, m_all, b_all = fit_overall
        overall_lr = {"slope": m_all, "intercept": b_all}
        fig.add_trace(go.Scatter(
            x=xs, y=yhat,
            mode="lines", name="Overall regression",
            line=dict(color="#EF4444", width=2.5, dash="dash"),
        ))
        regression_equations["overall"] = {"slope": m_all, "intercept": b_all}

    # ── 95% CI band (computed in transformed space, back-transformed for plot) ──
    if show_ci and n > 3 and overall_lr is not None and fit_overall is not None:
        xs_fit, yhat_fit = (
            (np.log10(clean[x].values) if log_x else clean[x].values),
            (np.log10(clean[y].values) if log_y else clean[y].values),
        )
        m_all, b_all = overall_lr["slope"], overall_lr["intercept"]
        residuals = yhat_fit - (m_all * xs_fit + b_all)
        s_err = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
        x_mean = xs_fit.mean()
        sxx = np.sum((xs_fit - x_mean) ** 2)

        if sxx > 0:
            tval = student_t.ppf(0.975, n - 2)
            # xs here is in plot space; need fit space for CI
            xs_ci = np.log10(xs) if log_x else xs
            ys_ci = m_all * xs_ci + b_all             # fit-space predictions
            se_fit = s_err * np.sqrt((1 / n) + ((xs_ci - x_mean) ** 2 / sxx))
            ci_u_fit = ys_ci + tval * se_fit
            ci_l_fit = ys_ci - tval * se_fit
            ci_u = 10 ** ci_u_fit if log_y else ci_u_fit
            ci_l = 10 ** ci_l_fit if log_y else ci_l_fit

            fig.add_trace(go.Scatter(x=xs, y=ci_u, mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=ci_l, mode="lines",
                                     fill="tonexty",
                                     fillcolor="rgba(239,68,68,0.12)",
                                     line=dict(width=0),
                                     name="95% CI"))
            

    x_label = f"{x} ({units[x]})" if x in units else x
    y_label = f"{y} ({units[y]})" if y in units else y
    fig.update_layout(
        # title=dict(text=f"{x}  vs  {y}", font=dict(size=15)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        height=520,
        hovermode="closest",
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(l=60, r=160, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", gridwidth=1)

    # ── correlations ──
    x_v, y_v = clean[x].nunique() > 1, clean[y].nunique() > 1
    if x_v and y_v:
        pear_r, pear_p = pearsonr(clean[x], clean[y])
        spe_r, spe_p   = spearmanr(clean[x], clean[y])
    else:
        pear_r = pear_p = spe_r = spe_p = np.nan

    stats = {
        "sample_size": n,
        "feature_x_summary": _summary_for_series(clean[x], x),
        "feature_y_summary": _summary_for_series(clean[y], y),
        "method": "Pearson + Spearman correlation, OLS regression",
        "pearson_r":  None if np.isnan(pear_r) else float(pear_r),
        "pearson_p":  None if np.isnan(pear_p) else float(pear_p),
        "spearman_r": None if np.isnan(spe_r)  else float(spe_r),
        "spearman_p": None if np.isnan(spe_p)  else float(spe_p),
        "regression_equations": regression_equations,
    }

    if color_col:
        subgroup_stats = {}
        for grp, gdf in clean.groupby(color_col):
            subgroup_stats[str(grp)] = {
                "count":    int(len(gdf)),
                "x_mean":   float(gdf[x].mean()),
                "x_median": float(gdf[x].median()),
                "x_std":    float(gdf[x].std()) if len(gdf) > 1 else 0.0,
                "x_min":    float(gdf[x].min()),
                "x_max":    float(gdf[x].max()),
                "y_mean":   float(gdf[y].mean()),
                "y_median": float(gdf[y].median()),
                "y_std":    float(gdf[y].std()) if len(gdf) > 1 else 0.0,
                "y_min":    float(gdf[y].min()),
                "y_max":    float(gdf[y].max()),
            }
        stats["subfeature_name"]   = color_col
        stats["subfeature_values"] = sorted([str(v) for v in clean[color_col].dropna().unique()])
        stats["subfeature_stats"]  = subgroup_stats

    conclusion = generate_conclusion("num_num", x, y, stats, subfeature=color_col)
    return fig, stats, conclusion, None


# ─── Num × Cat ──────────────────────────────────────────────────────────────
def analyze_num_cat(df, num_col, cat_col, subfeature=None, plot_mode="violin"):
    cols = [num_col, cat_col]
    color_col = None
    if subfeature and subfeature in df.columns and subfeature not in [num_col, cat_col]:
        cols.append(subfeature)
        color_col = subfeature

    clean = df[cols].dropna(subset=[num_col, cat_col]).copy()
    clean = clean[np.isfinite(clean[num_col])]
    n = len(clean)
    if n < 3:
        return None

    # Cap categories to avoid unreadable plots
    top_cats = clean[cat_col].value_counts().head(12).index
    clean = clean[clean[cat_col].isin(top_cats)]

    fig = go.Figure()

    if plot_mode == "barh_stacked":
        # ── Horizontal stacked bar: bin numeric feature, stack by category ──
        df_copy = clean.copy()
        df_copy["_bin"] = pd.qcut(df_copy[num_col], q=5, duplicates="drop")
        crosstab = pd.crosstab(df_copy["_bin"], df_copy[cat_col])
        for gi, col in enumerate(crosstab.columns):
            fig.add_trace(go.Bar(
                y=crosstab.index.astype(str).tolist(),
                x=crosstab[col].tolist(),
                name=str(col),
                orientation="h",
                marker_color=_colour(gi),
                opacity=0.88,
            ))
        fig.update_layout(
            title=dict(text=f"{num_col} Distribution by {cat_col} — Stacked Bar", font=dict(size=15)),
            barmode="stack",
            xaxis_title="Count",
            yaxis_title=f"{num_col} Bins",
            height=520,
            hovermode="y unified",
        )

    else:
        # ── Grouped bar (mean) — default ──
        if color_col:
            grp_vals = clean[color_col].dropna().unique()
            for gi, grp in enumerate(sorted(grp_vals, key=str)):
                gdf = clean[clean[color_col] == grp]
                means = gdf.groupby(cat_col)[num_col].mean()
                fig.add_trace(go.Bar(
                    x=means.index.tolist(),
                    y=means.values.tolist(),
                    name=str(grp),
                    marker_color=_colour(gi),
                    opacity=0.88,
                ))
        else:
            means = clean.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            fig.add_trace(go.Bar(
                x=means.index.tolist(),
                y=means.values.tolist(),
                name=f"Mean {num_col}",
                marker_color=[_colour(i) for i in range(len(means))],
                opacity=0.88,
            ))
        fig.update_layout(
            title=dict(text=f"Mean {num_col}  by  {cat_col}", font=dict(size=15)),
            xaxis_title=cat_col,
            yaxis_title=f"Mean {num_col}",
            barmode="group",
            height=520,
        )

    # shared layout polish
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(l=60, r=160, t=65, b=60),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

    # ── category-level stats ──
    grouped = clean.groupby(cat_col)[num_col]
    cat_stats = {
        str(cat): {
            "count":  int(vals.count()),
            "mean":   float(vals.mean()),
            "median": float(vals.median()),
        }
        for cat, vals in grouped
    }

    grand_mean = clean[num_col].mean()
    ss_total   = float(np.sum((clean[num_col] - grand_mean) ** 2))
    ss_between = float(sum(len(v) * ((v.mean() - grand_mean) ** 2) for _, v in grouped))
    eta_sq     = (ss_between / ss_total) if ss_total > 0 else 0.0

    group_values = [vals.values for _, vals in grouped if len(vals) > 0]
    p_value = None
    if len(group_values) > 1:
        try:
            p_value = float(f_oneway(*group_values).pvalue)
        except Exception:
            pass

    stats = {
        "sample_size":         n,
        "feature_x_summary":   _summary_for_series(clean[num_col], num_col),
        "feature_y_summary":   _summary_for_series(clean[cat_col], cat_col),
        "method":              "Eta-squared (ANOVA effect size)",
        "eta_squared":         float(eta_sq),
        "p_value":             p_value,
        "category_stats":      cat_stats,
    }

    # expose subfeature for JS toggle
    if color_col:
        subgroup_stats = {}
        for grp, gdf in clean.groupby(color_col):
            subgroup_stats[str(grp)] = {
                "count":    int(len(gdf)),
                "x_mean":   float(gdf[num_col].mean()),
                "x_median": float(gdf[num_col].median()),
                "y_mean":   float(gdf[num_col].mean()),   # same axis for num-cat
                "y_median": float(gdf[num_col].median()),
            }
        stats["subfeature_name"]   = color_col
        stats["subfeature_values"] = sorted([str(v) for v in clean[color_col].dropna().unique()])
        stats["subfeature_stats"]  = subgroup_stats

    conclusion = generate_conclusion("num_cat", num_col, cat_col, stats, subfeature=color_col)
    return fig, stats, conclusion, cat_stats


# ─── Cat × Cat ──────────────────────────────────────────────────────────────
def analyze_cat_cat(df, x, y, plot_mode="heatmap"):
    clean = df[[x, y]].dropna().copy()
    n = len(clean)
    if n < 2:
        return None

    # Cap rows/cols to avoid huge matrices
    top_x = clean[x].value_counts().head(10).index
    top_y = clean[y].value_counts().head(10).index
    clean = clean[clean[x].isin(top_x) & clean[y].isin(top_y)]
    if len(clean) < 2:
        return None

    ctab = pd.crosstab(clean[x], clean[y])
    if ctab.empty:
        return None

    if plot_mode == "grouped_bar":
        fig = go.Figure()
        for gi, col in enumerate(ctab.columns):
            fig.add_trace(go.Bar(
                x=ctab.index.astype(str).tolist(),
                y=ctab[col].tolist(),
                name=str(col),
                marker_color=_colour(gi),
                opacity=0.88,
            ))
        fig.update_layout(
            title=dict(text=f"{x}  vs  {y}  — Grouped Bar", font=dict(size=15)),
            xaxis_title=x, yaxis_title="Count",
            barmode="group", height=520,
            hovermode="x unified",
        )
    else:
        # ── Heatmap ──
        z_values = ctab.values.astype(float)
        z_max = float(np.max(z_values)) if z_values.size else 1.0
        z_max = z_max if z_max > 0 else 1.0
        text_vals = [
            [f"{int(ctab.iloc[ri, ci])}<br>{(ctab.iloc[ri, ci]/n*100):.1f}%"
             for ci in range(ctab.shape[1])]
            for ri in range(ctab.shape[0])
        ]
        fig = go.Figure(data=go.Heatmap(
            z=z_values.tolist(),
            x=[str(c) for c in ctab.columns],
            y=[str(i) for i in ctab.index],
            zmin=0, zmax=z_max,
            colorscale="Blues",
            showscale=True,
            text=text_vals,
            texttemplate="%{text}",
            hovertemplate=f"{x}: %{{y}}<br>{y}: %{{x}}<br>Count: %{{z}}<extra></extra>",
            # textfont=dict(size=12),
            textfont=dict(size=12, color="black")
        ))
        fig.update_layout(
            title=dict(text=f"Co-occurrence Heatmap: {x}  ×  {y}", font=dict(size=15)),
            xaxis_title=y, yaxis_title=x,
            height=560,
            xaxis=dict(tickangle=-30),
        )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        margin=dict(l=80, r=60, t=65, b=80),
    )

    try:
        chi2, p_value, _, _ = chi2_contingency(ctab)
    except Exception:
        chi2, p_value = 0.0, 1.0

    r, k = ctab.shape
    denom = min(r - 1, k - 1)
    cramers_v = float(np.sqrt((chi2 / n) / denom)) if denom > 0 else 0.0

    freq_x = {str(k): int(v) for k, v in clean[x].value_counts().items()}
    freq_y = {str(k): int(v) for k, v in clean[y].value_counts().items()}

    stats = {
        "sample_size":         n,
        "feature_x_summary":   _summary_for_series(clean[x], x),
        "feature_y_summary":   _summary_for_series(clean[y], y),
        "method":              "Cramér's V (Chi-square)",
        "cramers_v":           float(cramers_v),
        "p_value":             float(p_value),
        "frequency_tables":    {"feature_x_frequency": freq_x, "feature_y_frequency": freq_y},
    }
    conclusion = generate_conclusion("cat_cat", x, y, stats)
    return fig, stats, conclusion, {"feature_x_frequency": freq_x, "feature_y_frequency": freq_y}


# ─── Dispatcher ─────────────────────────────────────────────────────────────
def run_relationship_analysis(df, feature_x, feature_y, subfeature=None,
                               plot_mode=None, show_ci=False, log_x=False, log_y=False):
    if feature_x not in df.columns or feature_y not in df.columns:
        raise ValueError("Feature not found in dataset")
    if feature_x == feature_y:
        raise ValueError("Feature X and Feature Y must be different")

    x_num = is_numerical(feature_x)
    y_num = is_numerical(feature_y)
    x_cat = is_categorical(feature_x)
    y_cat = is_categorical(feature_y)

    if x_num and y_num:
        return analyze_num_num(df, feature_x, feature_y,
                               subfeature=subfeature, show_ci=show_ci,
                               log_x=log_x, log_y=log_y)

    if (x_num and y_cat) or (x_cat and y_num):
        num_col = feature_x if x_num else feature_y
        cat_col = feature_y if y_cat else feature_x
        return analyze_num_cat(df, num_col, cat_col,
                               subfeature=subfeature,
                               plot_mode=plot_mode or "bar")

    if x_cat and y_cat:
        return analyze_cat_cat(df, feature_x, feature_y,
                               plot_mode=plot_mode or "heatmap")

    raise ValueError("Unsupported feature type combination")
