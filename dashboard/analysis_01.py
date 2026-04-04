import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

# Feature classification
NUMERICAL_TRAITS = ["log_mass","mass","wing_len","tarsus",
                    "tail","beak_depth","beak_culmen","beak_nares","beak_width",
                    "hwi","kipps", "secondary", "wing_pointedness", "tail_to_wing"]

CATEGORICAL_TRAITS = ["habitat","habitat_density","migration","trophic_level","trophic_niche","lifestyle"]
ALL_TRAITS = NUMERICAL_TRAITS + CATEGORICAL_TRAITS





units = {"mass" : "gram"}

# Sub-feature options for grouping/coloring
SUBFEATURE_OPTIONS = {
    "order_birdlife": "Order",
    "habitat_density": "Habitat Density",
    "migration": "Migration",
    "trophic_niche": "Trophic Niche",
    "trophic_level": "Trophic Level",
}

# Filter options
FILTER_COLS = {
    "order_birdlife": "Order",
    "family_birdlife": "Family",
    "trophic_niche": "Trophic Niche",
    "habitat_density": "Habitat Density",
    "migration": "Migration",
}


def is_numerical(column):
    """Check if a column is numerical."""
    return column in NUMERICAL_TRAITS


def is_categorical(column):
    """Check if a column is categorical."""
    return column in CATEGORICAL_TRAITS


def get_filtered_data(data, filters):
    """Apply multiple filters to the dataset."""
    filtered = data.copy()
    for col, value in filters.items():
        if col == "lat_range" and isinstance(value, list) and len(value) == 2:
            filtered = filtered[
                (filtered["lat_centroid"] >= value[0]) & (filtered["lat_centroid"] <= value[1])
            ]
        elif col == "lon_range" and isinstance(value, list) and len(value) == 2:
            filtered = filtered[
                (filtered["lon_centroid"] >= value[0]) & (filtered["lon_centroid"] <= value[1])
            ]
        elif value:
            filtered = filtered[filtered[col] == value]
    return filtered


def plot_numerical_histogram_kde(data, column, use_log=False):
    """Plot histogram + KDE for numerical features."""
    clean_data = data[column].dropna()
    if clean_data.empty:
        x_label = f"{column} ({units[column]})" if column in units else column
        fig = go.Figure()
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=x_label,
            yaxis_title='Count',
            height=400
        )
        return fig.to_html(full_html=False)
    
    if use_log:
        clean_data = np.log10(clean_data + 1)  # Avoid log(0)
    
    # Create histogram with adaptive bins (denser than default for better KDE fit).
    if len(clean_data) > 1:
        raw_edges = np.histogram_bin_edges(clean_data, bins='fd')
        if len(raw_edges) - 1 < 40:
            raw_edges = np.linspace(clean_data.min(), clean_data.max(), 41)
        elif len(raw_edges) - 1 > 100:
            raw_edges = np.linspace(clean_data.min(), clean_data.max(), 101)
        bin_width = float(raw_edges[1] - raw_edges[0]) if len(raw_edges) > 1 else 1.0
        nbins = max(50, min(100, len(raw_edges) - 1))
    else:
        bin_width = 1.0
        nbins = 50

    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=clean_data,
        name='Histogram',
        marker_color='rgba(100, 150, 255, 0.7)',
        nbinsx=nbins
    ))
    # fig.update_layout(template="plotly_dark")
    # Add KDE curve
    try:
        kde = gaussian_kde(clean_data)
        x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
        kde_values = kde(x_range)
        # Scale density to histogram count-space so KDE is visually meaningful.
        kde_values = kde_values * len(clean_data) * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_values,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ))
    except:
        pass
    
    title = f"Distribution of {column}" + (" (log scale)" if use_log else "")
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title='Count',
        hovermode='x unified',
        height=400
    )
    
    return fig.to_html(full_html=False)


def plot_feature_by_subfeature(data, feature, subfeature, graph_type='bar'):
    """Plot bar chart or horizontal stacked bar for feature grouped by subfeature."""
    if feature == subfeature:
        return None, {}

    df = data[[feature, subfeature]].dropna()
    if df.empty:
        return None, {}

    if is_numerical(feature):
        if graph_type == 'bar':
            # Vertical bar of mean per subfeature group
            group_means = df.groupby(subfeature)[feature].mean().reset_index()
            group_counts = df.groupby(subfeature)[feature].count().reset_index()
            group_means['count'] = group_counts[feature]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=group_means[subfeature],
                y=group_means[feature],
                name='Mean',
                marker_color='rgba(100, 150, 255, 0.7)',
                text=group_means['count'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Mean {feature} by {subfeature}",
                xaxis_title=subfeature,
                yaxis_title=f'Mean {feature}',
                height=400
            )
            
            group_stats = {}
            for _, row in group_means.iterrows():
                group = row[subfeature]
                group_data = df[df[subfeature] == group][feature]
                group_stats[str(group)] = {
                    'count': int(len(group_data)),
                    'mean': float(group_data.mean()),
                    'median': float(group_data.median()),
                    'min': float(group_data.min()),
                    'max': float(group_data.max()),
                    'q1': float(group_data.quantile(0.25)),
                    'q3': float(group_data.quantile(0.75))
                }
            
        elif graph_type == 'barh_stacked':
            # Horizontal stacked bar: bin the numerical feature and stack by subfeature
            # Bin the feature into quantiles
            df_copy = df.copy()
            df_copy['bin'] = pd.qcut(df_copy[feature], q=5, duplicates='drop')
            crosstab = pd.crosstab(df_copy['bin'], df_copy[subfeature])
            
            fig = go.Figure()
            for sub in crosstab.columns:
                fig.add_trace(go.Bar(
                    y=crosstab.index.astype(str),
                    x=crosstab[sub],
                    name=str(sub),
                    orientation='h'
                ))
            
            fig.update_layout(
                title=f"{feature} Distribution by {subfeature}",
                barmode='stack',
                height=400,
                yaxis_title=f'{feature} Bins',
                xaxis_title='Count',
                hovermode='y unified'
            )
            
            group_stats = {}
            for group in df[subfeature].unique():
                group_data = df[df[subfeature] == group][feature]
                group_stats[str(group)] = {
                    'count': int(len(group_data)),
                    'mean': float(group_data.mean()),
                    'median': float(group_data.median()),
                    'min': float(group_data.min()),
                    'max': float(group_data.max()),
                    'q1': float(group_data.quantile(0.25)),
                    'q3': float(group_data.quantile(0.75))
                }
        else:
            return None, {}

    elif is_categorical(feature):
        if graph_type == 'bar':
            # Vertical bar of counts per category
            counts = df[feature].value_counts().reset_index()
            counts.columns = [feature, 'count']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=counts[feature],
                y=counts['count'],
                marker_color='rgba(100, 150, 255, 0.7)'
            ))
            
            fig.update_layout(
                title=f"Distribution of {feature}",
                xaxis_title=feature,
                yaxis_title='Count',
                height=400
            )
            
            group_stats = {}
            for cat in counts[feature]:
                group_data = df[df[feature] == cat][subfeature] if subfeature else df[df[feature] == cat]
                group_stats[str(cat)] = {'count': int(counts[counts[feature] == cat]['count'].iloc[0])}
        
        elif graph_type == 'barh_stacked':
            # Horizontal stacked bar
            crosstab = pd.crosstab(df[feature], df[subfeature])
            
            fig = go.Figure()
            for sub in crosstab.columns:
                fig.add_trace(go.Bar(
                    y=crosstab.index,
                    x=crosstab[sub],
                    name=str(sub),
                    orientation='h'
                ))
            
            fig.update_layout(
                title=f"{feature} by {subfeature}",
                barmode='stack',
                height=400,
                yaxis_title=feature,
                xaxis_title='Count',
                hovermode='y unified'
            )
            
            group_stats = {}
            for cat in df[feature].unique():
                group_stats[str(cat)] = {'count': int(len(df[df[feature] == cat]))}
        else:
            return None, {}

    return fig.to_dict(), group_stats


def plot_categorical_bar(data, column):
    """Plot horizontal bar chart for categorical features."""
    counts = data[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(1)
    
    fig = px.bar(
        counts,
        y=column,
        x='count',
        orientation='h',
        title=f"Distribution of {column}",
        hover_data={'percentage': True}
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='y unified'
    )
    
    return fig.to_html(full_html=False)


def plot_categorical_stacked_bar(data, column, subfeature):
    """Plot stacked bar chart for categorical features grouped by sub-feature."""
    crosstab = pd.crosstab(data[column], data[subfeature])
    
    fig = go.Figure()
    
    for sub in crosstab.columns:
        fig.add_trace(go.Bar(
            y=crosstab.index,
            x=crosstab[sub],
            name=str(sub),
            orientation='h'
        ))
    
    fig.update_layout(
        title=f"{column} by {subfeature}",
        barmode='stack',
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='y unified'
    )
    
    return fig.to_html(full_html=False)


def get_numerical_stats(data, column):
    """Calculate stats for numerical features."""
    clean_data = data[column].dropna()
    
    stats = {
        'Mean': f"{clean_data.mean():.2f}",
        'Median': f"{clean_data.median():.2f}",
        'Min': f"{clean_data.min():.2f}",
        'Max': f"{clean_data.max():.2f}",
        'Q1': f"{clean_data.quantile(0.25):.2f}",
        'Q3': f"{clean_data.quantile(0.75):.2f}",
        'Count': f"{len(clean_data)}"
    }
    return stats


def get_categorical_stats(data, column):
    """Calculate stats for categorical features."""
    counts = data[column].value_counts()
    missing = data[column].isna().sum()
    missing_pct = (missing / len(data) * 100) if len(data) > 0 else 0
    
    stats = {
        'Unique': f"{len(counts)}",
        'Mode': f"{counts.index[0] if len(counts) > 0 else 'N/A'}",
        'Missing %': f"{missing_pct:.1f}%",
        'Total': f"{len(data)}"
    }
    return stats


def get_top_categories(data, column, n=10):
    """Get top N categories."""
    counts = data[column].value_counts().head(n)
    return {str(cat): int(count) for cat, count in counts.items()}


def compute_correlations(data):
    """Pre-compute correlations for numerical features and chi-square for categorical."""
    numerical_cols = [col for col in NUMERICAL_TRAITS if col in data.columns]
    categorical_cols = [col for col in CATEGORICAL_TRAITS if col in data.columns]
    
    # Numerical correlations
    num_corr = {}
    if numerical_cols:
        corr_matrix = data[numerical_cols].corr()
        for col1 in numerical_cols:
            num_corr[col1] = {}
            for col2 in numerical_cols:
                if col1 != col2:
                    num_corr[col1][col2] = abs(corr_matrix.loc[col1, col2])
    
    # Categorical chi-square
    from scipy.stats import chi2_contingency
    cat_chi2 = {}
    for col1 in categorical_cols:
        cat_chi2[col1] = {}
        for col2 in categorical_cols:
            if col1 != col2:
                try:
                    contingency = pd.crosstab(data[col1], data[col2])
                    if contingency.size < 10000:  # Limit to avoid slow computation
                        chi2, p, _, _ = chi2_contingency(contingency)
                        cat_chi2[col1][col2] = chi2
                    else:
                        cat_chi2[col1][col2] = 0
                except:
                    cat_chi2[col1][col2] = 0
    
    return num_corr, cat_chi2


def get_top_related_features(feature, num_corr, cat_chi2, n=5):
    """Get top N related features for a given feature."""
    if is_numerical(feature):
        if feature in num_corr:
            related = sorted(num_corr[feature].items(), key=lambda x: x[1], reverse=True)[:n]
            return [{'feature': f, 'score': s, 'type': 'correlation'} for f, s in related]
    elif is_categorical(feature):
        if feature in cat_chi2:
            related = sorted(cat_chi2[feature].items(), key=lambda x: x[1], reverse=True)[:n]
            return [{'feature': f, 'score': s, 'type': 'chi_square'} for f, s in related]
    return []