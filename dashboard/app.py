from flask import Flask, render_template, request, jsonify
import json
from analysis_01 import (
    is_numerical, is_categorical, get_filtered_data,
    plot_numerical_histogram_kde, plot_feature_by_subfeature,
    plot_categorical_bar, plot_categorical_stacked_bar,
    get_numerical_stats, get_categorical_stats, get_top_categories,
    NUMERICAL_TRAITS, CATEGORICAL_TRAITS, SUBFEATURE_OPTIONS, FILTER_COLS
)
from analysis_02 import run_relationship_analysis
from analysis_03 import (
    RADAR_FEATURES_DEFAULT,
    build_species_table,
    get_numeric_columns,
    precompute_normalized_numeric,
    euclidean_distance_normed,
    load_tree,
    phylo_distance,
    row_to_jsonable_dict,
)
from analysis_04 import rank_species_by_traits
import pandas as pd
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# Load dataset at startup
app.data = pd.read_csv('../data/processed/avonet_FE_01.csv')

# Section 3 precomputations (no per-request normalization)
app.species_table = build_species_table(app.data, species_col="species_birdtree")
app.species_list = sorted(app.species_table.index.tolist())
app.numeric_cols = get_numeric_columns(app.species_table, exclude=set())
app.norm_numeric, app.norm_meta = precompute_normalized_numeric(app.species_table, app.numeric_cols)
app.radar_features = [f for f in RADAR_FEATURES_DEFAULT if f in app.species_table.columns]

# Load phylogeny tree (optional but fast per-request distance)
try:
    app.phylo_tree, app.phylo_terminals = load_tree("../data/processed/Phylogeny.tre")
except Exception:
    app.phylo_tree, app.phylo_terminals = None, set()


def _to_json_safe_plotly(obj):
    """Convert Plotly payloads to JSON-serializable dict/list values."""
    return json.loads(json.dumps(obj, cls=PlotlyJSONEncoder))

@app.route('/')
def hello():
    return render_template('index.html', active_section=0)

@app.route('/section1', methods=['GET', 'POST'])
def section1():
    # Support potential POST requests gracefully (from old form behavior)
    return render_template(
        'section1.html',
        filter_options=FILTER_COLS,
        subfeature_options=SUBFEATURE_OPTIONS,
        numerical_traits=NUMERICAL_TRAITS,
        categorical_traits=CATEGORICAL_TRAITS,
        active_section=1,
    )


@app.route('/section2', methods=['GET'])
def section2():
    return render_template(
        'section2.html',
        filter_options=FILTER_COLS,
        subfeature_options=SUBFEATURE_OPTIONS,
        numerical_traits=NUMERICAL_TRAITS,
        categorical_traits=CATEGORICAL_TRAITS,
        active_section=2,
    )


@app.route('/section3', methods=['GET'])
def section3():
    return render_template('section3.html', active_section=3)


@app.route('/section4', methods=['GET'])
def section4():
    return render_template('section4.html', active_section=4)

@app.route('/api/analysis', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return jsonify({'error': 'Use POST with JSON body to call analysis endpoint'}), 405

    # POST behavior continues below
    """API endpoint for analysis plots and stats."""
    try:
        feature = request.json.get('feature')
        subfeature = request.json.get('subfeature')
        use_log = request.json.get('use_log', False)
        graph_type = request.json.get('graph_type', 'bar')
        filters = request.json.get('filters', {})
        
        # Get filtered data
        filtered_data = get_filtered_data(app.data, filters)
        filtered_count = len(filtered_data)
        total_count = len(app.data)
        
        if filtered_count == 0:
            return jsonify({
                'filtered_count': filtered_count,
                'total_count': total_count,
                'filters': filters,
                'no_data': True,
                'message': 'There is no species for selected filters.',
                'plot1': None,
                'plot2': None,
                'plot2_figure': None,
                'plot2_stats': None,
                'stats': {},
                'top_categories': None,
                'top_related': []
            })
        
        result = {
            'filtered_count': filtered_count,
            'total_count': total_count,
            'filters': filters,
            'no_data': False
        }

        # Prevent duplicate feature/subfeature selection from breaking grouped logic.
        if subfeature == feature:
            subfeature = None
        
        # Determine if feature is numerical or categorical
        if is_numerical(feature):
            # Plot 1: Histogram + KDE
            result['plot1'] = plot_numerical_histogram_kde(filtered_data, feature, use_log=use_log)
            
            # Plot 2: Bar plot grouped by subfeature
            if subfeature and subfeature in filtered_data.columns:
                scatter_payload, group_stats = plot_feature_by_subfeature(filtered_data, feature, subfeature, graph_type)
                result['plot2'] = None
                result['plot2_figure'] = _to_json_safe_plotly(scatter_payload)
                result['plot2_stats'] = group_stats
            else:
                result['plot2'] = None
                result['plot2_figure'] = None
                result['plot2_stats'] = None
            
            # Stats
            result['stats'] = get_numerical_stats(filtered_data, feature)
            result['top_categories'] = None
            # Correlation/chi-square output removed for performance.
            result['top_related'] = []
        
        elif is_categorical(feature):
            # Plot 1: Horizontal Bar
            result['plot1'] = plot_categorical_bar(filtered_data, feature)
            
            # Plot 2: Bar plot grouped by subfeature
            if subfeature and subfeature in filtered_data.columns:
                scatter_payload, group_stats = plot_feature_by_subfeature(filtered_data, feature, subfeature, graph_type)
                result['plot2'] = None
                result['plot2_figure'] = _to_json_safe_plotly(scatter_payload)
                result['plot2_stats'] = group_stats
            else:
                result['plot2'] = None
                result['plot2_figure'] = None
                result['plot2_stats'] = None
            
            stats = get_categorical_stats(filtered_data, feature)
            top_cats = get_top_categories(filtered_data, feature)
            result['stats'] = stats
            result['top_categories'] = top_cats
            # Correlation/chi-square output removed for performance.
            result['top_related'] = []
        
        else:
            return jsonify({'error': 'Feature not found or invalid'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis2', methods=['POST'])
def analyze_section2():
    try:
        feature_x = request.json.get('feature_x')
        feature_y = request.json.get('feature_y')
        subfeature = request.json.get('subfeature')
        filters = request.json.get('filters', {})
        plot_mode = request.json.get('plot_mode')
        show_ci = request.json.get('show_ci', False)
        log_x = request.json.get('log_x', False)
        log_y = request.json.get('log_y', False)

        filtered_data = get_filtered_data(app.data, filters)
        filtered_count = len(filtered_data)
        total_count = len(app.data)

        if filtered_count == 0:
            return jsonify({
                'filtered_count': filtered_count,
                'total_count': total_count,
                'filters': filters,
                'no_data': True,
                'message': 'There is no species for selected filters.',
            })

        result = run_relationship_analysis(
            filtered_data,
            feature_x=feature_x,
            feature_y=feature_y,
            subfeature=subfeature,
            plot_mode=plot_mode,
            show_ci=show_ci,
            log_x=log_x,
            log_y=log_y,
        )

        if result is None:
            return jsonify({
                'filtered_count': filtered_count,
                'total_count': total_count,
                'no_data': True,
                'message': 'Not enough data after dropping missing values.',
            })

        fig, stats, conclusion, _ = result
        return jsonify({
            'filtered_count': filtered_count,
            'total_count': total_count,
            'no_data': False,
            'plot': _to_json_safe_plotly(fig.to_dict()),
            'stats': _to_json_safe_plotly(stats),
            'conclusion': conclusion,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/species-search', methods=['GET'])
def species_search():
    q = (request.args.get("q") or "").strip().lower()
    try:
        offset = int(request.args.get("offset") or 0)
        limit = int(request.args.get("limit") or 50)
    except Exception:
        offset, limit = 0, 50
    limit = max(5, min(200, limit))
    offset = max(0, offset)

    # If query is empty, return an initial slice (for scrollable browsing).
    if q == "":
        vals = app.species_list[offset: offset + limit]
        return jsonify({"values": vals, "offset": offset, "limit": limit, "total": len(app.species_list)})

    if len(q) < 2:
        return jsonify({"values": [], "offset": offset, "limit": limit, "total": 0})

    matches = [s for s in app.species_list if q in s.lower()]
    vals = matches[offset: offset + limit]
    return jsonify({"values": vals, "offset": offset, "limit": limit, "total": len(matches)})


@app.route('/api/analysis3', methods=['POST'])
def analyze_section3():
    try:
        species = request.json.get("species", [])
        if not isinstance(species, list):
            return jsonify({"error": "species must be a list"}), 400
        species = [str(s).strip() for s in species if str(s).strip()]
        # keep order, unique
        seen = set()
        species = [s for s in species if not (s in seen or seen.add(s))]

        if len(species) == 0:
            return jsonify({"rows": [], "distances": [], "radar_features": [], "radar_series": [], "conclusion": ""})

        missing = [s for s in species if s not in app.species_table.index]
        if missing:
            return jsonify({"error": f"Species not found: {', '.join(missing[:5])}"}), 400

        rows = [row_to_jsonable_dict(app.species_table.loc[s]) for s in species]

        # Pairwise distances for up to 3 explicitly, and for more as well
        distances = []
        for i in range(len(species)):
            for j in range(i + 1, len(species)):
                a, b = species[i], species[j]
                euc, nfeat = euclidean_distance_normed(app.norm_numeric, a, b)
                phy = phylo_distance(app.phylo_tree, app.phylo_terminals, a, b)
                distances.append({"a": a, "b": b, "euclidean": euc, "euclidean_n": nfeat, "phylo": phy})

        radar_features = app.radar_features
        radar_series = []
        if len(species) >= 2 and radar_features:
            for s in species:
                vals = []
                r = app.species_table.loc[s]
                for f in radar_features:
                    v = r.get(f)
                    vals.append(None if pd.isna(v) else float(v))
                radar_series.append({"name": s, "values": vals})

        # Geo points (lat/lon centroid) if available
        geo_points = []
        geo_missing = []
        for s in species:
            r = app.species_table.loc[s]
            lat = r.get("lat_centroid")
            lon = r.get("lon_centroid")
            if pd.isna(lat) or pd.isna(lon):
                geo_missing.append(s)
                continue
            geo_points.append({"name": s, "lat": float(lat), "lon": float(lon)})

        # Rich data-driven conclusion
        conclusion = ""
        if len(species) == 1:
            sp = species[0]
            row = app.species_table.loc[sp]
            num_cols_avail = [c for c in app.numeric_cols if not pd.isna(row.get(c))]
            conclusion = (
                f"**{sp}** has been profiled across {len(num_cols_avail)} numeric traits "
                "in the AVONET dataset. "
                "The row table above shows its complete morphological and ecological record "
                "including beak, wing, tarsus, tail, and mass measurements. "
                "Compare it with other species by adding them below to reveal how unusual or typical "
                "this species is within its ecological guild."
            )
        elif len(species) == 2:
            a, b = species[0], species[1]
            best_dist = distances[0] if distances else {}
            euc = best_dist.get("euclidean")
            phy = best_dist.get("phylo")
            euc_str = f"{euc:.3f}" if isinstance(euc, float) else "N/A"
            phy_str = f"{phy:.1f} branches" if isinstance(phy, float) else "N/A"
            # determine which numeric traits differ most
            diff_traits = []
            for col in app.numeric_cols:
                va = app.species_table.loc[a].get(col)
                vb = app.species_table.loc[b].get(col)
                if pd.notna(va) and pd.notna(vb) and va != 0:
                    diff_traits.append((col, abs(float(va) - float(vb)) / (abs(float(va)) + 1e-9)))
            diff_traits.sort(key=lambda x: x[1], reverse=True)
            top_diffs = ", ".join(t[0] for t in diff_traits[:3]) if diff_traits else "multiple traits"
            conclusion = (
                f"**{a}** and **{b}** have a normalised Euclidean trait distance of **{euc_str}** "
                f"and a phylogenetic distance of **{phy_str}**. "
                f"The largest proportional differences are in **{top_diffs}**, "
                "indicating these are the primary axes of morphological divergence between the two species. "
                "A small Euclidean distance alongside a large phylogenetic distance would suggest "
                "convergent evolution; the opposite pattern would imply conserved ancestral morphology."
            )
        else:
            # multiple species
            all_euc = [d["euclidean"] for d in distances if isinstance(d.get("euclidean"), float)]
            min_euc = min(all_euc) if all_euc else None
            max_euc = max(all_euc) if all_euc else None
            closest_pair = min(distances, key=lambda d: d.get("euclidean") or 9999) if distances else {}
            farthest_pair = max(distances, key=lambda d: d.get("euclidean") or -1) if distances else {}
            close_str = (
                f"'{closest_pair.get('a')}' ↔ '{closest_pair.get('b')}' "
                f"(d = {closest_pair['euclidean']:.3f})"
                if closest_pair.get("euclidean") is not None else "N/A"
            )
            far_str = (
                f"'{farthest_pair.get('a')}' ↔ '{farthest_pair.get('b')}' "
                f"(d = {farthest_pair['euclidean']:.3f})"
                if farthest_pair.get("euclidean") is not None else "N/A"
            )
            conclusion = (
                f"Across {len(species)} selected species, normalised Euclidean distances range from "
                f"**{min_euc:.3f}** to **{max_euc:.3f}**. "
                f"The morphologically closest pair is {close_str}, while the most divergent is {far_str}. "
                "The radar chart shows how each species positions itself across the six core morphological axes — "
                "overlapping regions indicate shared trait dimensions while non-overlapping arcs reveal ecological specialisation. "
                "Phylogenetic distance (where available) contextualises whether morphological similarity arises from "
                "shared ancestry or independent convergent evolution."
            )

        return jsonify({
            "rows": rows,
            "distances": distances,
            "radar_features": radar_features,
            "radar_series": radar_series,
            "geo_points": geo_points,
            "geo_missing": geo_missing,
            "conclusion": conclusion,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/feature-options', methods=['GET'])
def feature_options():
    # Section 4: numeric traits only (as defined in Section 1).
    allowed_numeric = [c for c in NUMERICAL_TRAITS if c in app.species_table.columns]
    return jsonify({
        "numerical": allowed_numeric,
        "categorical": [],
        "all": allowed_numeric,
    })


@app.route('/api/analysis4', methods=['POST'])
def analyze_section4():
    try:
        traits = request.json.get("traits", [])
        target_feature = request.json.get("target_feature")
        top_k = int(request.json.get("top_k", 10))
        top_k = max(1, min(30, top_k))

        allowed_numeric = [c for c in NUMERICAL_TRAITS if c in app.species_table.columns]
        ranked = rank_species_by_traits(
            species_table=app.species_table,
            numeric_cols=allowed_numeric,
            norm_meta=app.norm_meta,
            trait_inputs=traits,
            top_k=top_k,
        )

        results = []
        for species_name, dist, used in ranked:
            row = app.species_table.loc[species_name]
            obj = {
                "species_birdtree": species_name,
                "distance": dist,
                "numeric_features_used": used,
            }
            if target_feature and target_feature in app.species_table.columns:
                v = row.get(target_feature)
                obj["target_feature_value"] = None if pd.isna(v) else (float(v) if isinstance(v, (int, float)) else str(v))
            results.append(obj)

        # light human text
        conclusion = ""
        if len(results) == 0:
            conclusion = "No probable species matched the entered trait pattern. Try relaxing categorical constraints or adding more numeric traits."
        else:
            conclusion = (
                "These are the closest species to your entered numeric trait pattern using normalized Euclidean similarity. "
                "Higher-ranked species are more similar across the numeric traits you provided."
            )

        return jsonify({
            "results": results,
            "target_feature": target_feature,
            "conclusion": conclusion,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/map-data', methods=['POST'])
def map_data():
    """Return lat/lon points for map, colored by a categorical column."""
    try:
        color_col = request.json.get('color_col', 'trophic_niche')
        filters = request.json.get('filters', {})
        filtered = get_filtered_data(app.data, filters)
        cols = ['lat_centroid', 'lon_centroid', 'species_birdtree']
        if color_col and color_col in filtered.columns:
            cols.append(color_col)
        subset = filtered[cols].dropna(subset=['lat_centroid', 'lon_centroid'])
        points = []
        for _, row in subset.iterrows():
            pt = {
                'lat': float(row['lat_centroid']),
                'lon': float(row['lon_centroid']),
                'species': str(row['species_birdtree']),
            }
            if color_col and color_col in row.index:
                pt['color_val'] = str(row[color_col]) if pd.notna(row[color_col]) else 'Unknown'
            points.append(pt)
        return jsonify({'points': points, 'color_col': color_col})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter-values/<filter_col>' )
def get_filter_values(filter_col):
    """Get unique values for a filter column."""
    try:
        if filter_col in app.data.columns:
            values = sorted(app.data[filter_col].dropna().unique().tolist())
            return jsonify({'values': values})
        else:
            return jsonify({'error': 'Column not found'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()