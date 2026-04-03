import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import math
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Make the dashboard utilize the full width of the monitor
st.set_page_config(page_title="Avian Explorer", layout="wide")

# ==========================================
# 1. DATA LOADING & STATE MANAGEMENT
# ==========================================
@st.cache_data
def load_clean_data():
    # Added utf-8-sig to strip invisible characters that cause KeyError on avibase_id
    return pd.read_csv('data/raw/avonet_FE_01.csv', encoding='utf-8-sig') 

@st.cache_data
def load_raw_data():
    return pd.read_csv('data/raw/avonet_cleaned.csv', encoding='utf-8-sig')

@st.cache_data
def load_image_data():
    return pd.read_csv('data/raw/bird_image_links.csv', encoding='utf-8-sig')

df_clean = load_clean_data()
df_raw = load_raw_data()
image_df = load_image_data()

if 'tab2_results' not in st.session_state:
    st.session_state['tab2_results'] = None
if 'working_df' not in st.session_state:
    st.session_state['working_df'] = df_clean

st.sidebar.header("📁 Data Source Manager")
st.sidebar.write("Select the dataset to use for Analysis:")

data_source = st.sidebar.radio(
    "Active Dataset:",
    [
        "🧪 Cleaned & Engineered Data", 
        "⚠️ Raw Data (With Missing Values)", 
        "🔍 Tab 2 Filtered Results", 
        "📂 Upload Custom CSV"
    ]
)

if data_source == "🧪 Cleaned & Engineered Data":
    st.session_state['working_df'] = df_clean
elif data_source == "⚠️ Raw Data (With Missing Values)":
    st.session_state['working_df'] = df_raw
    st.sidebar.info("💡 **Notice:** You are using uncleaned data. PCA clusters may shift as the algorithm drops rows with NaNs.")
elif data_source == "🔍 Tab 2 Filtered Results":
    if st.session_state['tab2_results'] is not None:
        st.session_state['working_df'] = st.session_state['tab2_results']
    else:
        st.sidebar.warning("⚠️ No filtered results yet. Run a search in Tab 2 first!")
        st.session_state['working_df'] = df_clean
elif data_source == "📂 Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your own dataset (.csv)", type=['csv'])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.session_state['working_df'] = custom_df
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error("Error reading the CSV file.")

st.title("🦅 Avian Trait Database Explorer")

# ==========================================
# 2. THE DYNAMIC SCHEMA MAPPER
# ==========================================
# This automatically figures out what the columns are named in the currently selected dataset!
active_df = st.session_state['working_df']

cols = {
    'id': 'avibase_id' if 'avibase_id' in active_df.columns else ('Avibase.ID' if 'Avibase.ID' in active_df.columns else active_df.columns[0]),
    'mass': 'mass' if 'mass' in active_df.columns else ('mass_avg' if 'mass_avg' in active_df.columns else 'Mass'),
    'wing': 'wing_len' if 'wing_len' in active_df.columns else ('wing_len_avg' if 'wing_len_avg' in active_df.columns else 'Wing.Length'),
    'beak': 'beak_culmen' if 'beak_culmen' in active_df.columns else ('beak_culmen_avg' if 'beak_culmen_avg' in active_df.columns else 'Beak.Length_Culmen'),
    'tarsus': 'tarsus' if 'tarsus' in active_df.columns else ('tarsus_avg' if 'tarsus_avg' in active_df.columns else 'Tarsus.Length'),
    'lat': 'lat_centroid' if 'lat_centroid' in active_df.columns else 'Centroid.Latitude',
    'lon': 'lon_centroid' if 'lon_centroid' in active_df.columns else 'Centroid.Longitude',
    'range': 'range_size' if 'range_size' in active_df.columns else 'Range.Size'
}

# ==========================================
# 3. HELPER MATH FUNCTIONS
# ==========================================
def calculate_overlap_percentage(lat1, lon1, r_user, lat2, lon2, area_bird):
    R_earth = 6371.0 
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R_earth * c
    
    r_bird = np.sqrt(area_bird / np.pi)
    
    if d >= (r_user + r_bird): return 0.0 
    elif d <= abs(r_user - r_bird): return (np.pi * min(r_user, r_bird)**2 / area_bird) * 100
    else:
        part1 = r_user**2 * np.arccos((d**2 + r_user**2 - r_bird**2) / (2 * d * r_user))
        part2 = r_bird**2 * np.arccos((d**2 + r_bird**2 - r_user**2) / (2 * d * r_bird))
        part3 = 0.5 * np.sqrt((-d + r_user + r_bird) * (d - r_user + r_bird) * (d + r_user - r_bird) * (d + r_user + r_bird))
        return ((part1 + part2 - part3) / area_bird) * 100

def generate_map_circle(lat, lon, radius_km, num_points=64):
    R = 6371.0 
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    d = radius_km / R
    circle_lats, circle_lons = [], []
    for i in range(num_points + 1):
        bearing = math.radians((i / num_points) * 360)
        new_lat = math.asin(math.sin(lat_rad) * math.cos(d) + math.cos(lat_rad) * math.sin(d) * math.cos(bearing))
        new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat_rad), math.cos(d) - math.sin(lat_rad) * math.sin(new_lat))
        circle_lats.append(math.degrees(new_lat))
        circle_lons.append(math.degrees(new_lon))
    return circle_lats, circle_lons

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Trait Lookup", "🎯 Reverse Matcher", "🧬 Trait Relationship Analysis and PCA", "🌍 Global Map"])

# ==========================================
# TAB 1: BIRD TRAIT LOOKUP & RELATIONSHIPS
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

    # --- 1. CREATE FAST LOOKUP DICTIONARIES (Bulletproof Method) ---
    
    # 1a. Dynamically find the ID column name
    if 'avibase_id' in active_df.columns:
        id_col = 'avibase_id'
    elif 'Avibase.ID' in active_df.columns:
        id_col = 'Avibase.ID'
    else:
        id_col = active_df.columns[0] # Ultimate fallback

    # Safe Lookup Dictionaries
    safe_lib1 = active_df['species_birdlife'].fillna("Name Unavailable") if 'species_birdlife' in active_df.columns else active_df[cols['id']]
    safe_lib2 = active_df['species_birdtree'].fillna("Name Unavailable") if 'species_birdtree' in active_df.columns else active_df[cols['id']]
    
    id_to_lib1 = dict(zip(active_df[cols['id']], safe_lib1))
    id_to_lib2 = dict(zip(active_df[cols['id']], safe_lib2))
    bird_ids = active_df[cols['id']].unique().tolist()

    # --- THE PHANTOM BIRD FIX ---
    # If the user filtered the dataset and the currently saved bird is no longer in it,
    # we MUST reset the memory to the first available bird in the new filtered list!
    if st.session_state.get('current_bird_id') not in bird_ids:
        if len(bird_ids) > 0:
            st.session_state['current_bird_id'] = bird_ids[0]
        else:
            st.warning("⚠️ No birds match your current filters! Go back to Tab 2 and broaden your search.")
            st.stop() # Stops drawing the rest of Tab 1 safely without crashing

    def update_from_lib1():
        st.session_state['current_bird_id'] = st.session_state['lib1_selector']
        st.session_state['lib2_selector'] = st.session_state['lib1_selector']
        
    def update_from_lib2():
        st.session_state['current_bird_id'] = st.session_state['lib2_selector']
        st.session_state['lib1_selector'] = st.session_state['lib2_selector']

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.selectbox("Search by BirdLife Name:", options=bird_ids, format_func=lambda x: str(id_to_lib1.get(x, "Unknown")), key='lib1_selector', on_change=update_from_lib1)
    with search_col2:
        st.selectbox("Search by BirdTree Name:", options=bird_ids, format_func=lambda x: str(id_to_lib2.get(x, "Unknown")), key='lib2_selector', on_change=update_from_lib2)

    st.write("---")

    selected_id = st.session_state['current_bird_id']
    bird_data = active_df[active_df[id_col] == selected_id]
    
    profile_col1, profile_col2 = st.columns([1, 2])
    
    with profile_col1:
        display_name = id_to_lib1.get(selected_id, "Unknown Species")
        if display_name == "Name Unavailable": display_name = id_to_lib2.get(selected_id, "Unknown Species")
            
        if 'avibase_id' in image_df.columns:
            matching_images = image_df[image_df['avibase_id'] == selected_id]
        else:
            matching_images = image_df[image_df['scientific_name'] == display_name]
            
        if not matching_images.empty:
            st.image(matching_images['image_url'].iloc[0], caption=display_name, use_container_width=True)
        else:
            st.info("📷 No image available.")
            
    with profile_col2:
        st.subheader(f"Data for: {display_name}")
        st.write("### Key Measurements")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            if cols['mass'] in active_df.columns and not pd.isna(bird_data[cols['mass']].iloc[0]):
                st.metric(label="Body Mass (g)", value=round(bird_data[cols['mass']].iloc[0], 2)) 
        with m_col2:
            if cols['wing'] in active_df.columns and not pd.isna(bird_data[cols['wing']].iloc[0]):
                st.metric(label="Wing Length (mm)", value=round(bird_data[cols['wing']].iloc[0], 2))
        with m_col3:
            if cols['beak'] in active_df.columns and not pd.isna(bird_data[cols['beak']].iloc[0]):
                st.metric(label="Beak Length (mm)", value=round(bird_data[cols['beak']].iloc[0], 2))
                
        st.write("") 
        st.subheader("Raw Data Row")
        st.dataframe(bird_data)


# ==========================================
# TAB 2: REVERSE TRAIT MATCHER
# ==========================================
with tab2:
    st.header("🧬 Reverse Trait Matcher")
    
    display_name = 'species_birdlife' if 'species_birdlife' in active_df.columns else cols['id']

    st.subheader("🌍 1. Geographic Filter (Optional)")
    use_map_filter = st.checkbox("🔍 Enable Geographic Search")
    
    if use_map_filter:
        col_map1, col_map2 = st.columns([1, 2])
        with col_map1:
            target_lat = st.slider("Target Latitude", min_value=-90.0, max_value=90.0, value=20.0, step=0.5)
            target_lon = st.slider("Target Longitude", min_value=-180.0, max_value=180.0, value=75.0, step=0.5)
            search_radius = st.slider("Search Radius (km)", min_value=100, max_value=5000, value=1000, step=100)
            min_overlap = st.slider("Min. Bird Range Overlap (%)", min_value=1, max_value=100, value=50)
            
        with col_map2:
            circle_lats, circle_lons = generate_map_circle(target_lat, target_lon, search_radius)
            fig_preview = go.Figure(go.Scattergeo())
            fig_preview.add_trace(go.Scattergeo(lat=circle_lats, lon=circle_lons, mode='lines', line=dict(color='red', width=2), fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', name='Search Area'))
            fig_preview.update_geos(center=dict(lat=target_lat, lon=target_lon), projection_scale=3, showcountries=True, showcoastlines=True, countrycolor="LightGrey")
            fig_preview.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_preview, use_container_width=True)

    st.write("Input specific measurements to find candidate species. Leave a trait unchecked to ignore it.")

    # DYNAMIC SEARCHABLE TRAITS
    searchable_traits = [cols['mass'], cols['wing'], cols['beak'], cols['tarsus']]
    active_filters = {}

    with st.form("reverse_matcher_form"):
        for trait in searchable_traits:
            if trait in active_df.columns:
                with st.container():
                    if st.checkbox(f"🔍 Filter by {trait}"):
                        c1, c2 = st.columns(2)
                        with c1: target_val = st.number_input(f"Target {trait}", min_value=0.0, value=50.0, step=1.0, key=f"target_{trait}")
                        with c2: tolerance = st.slider(f"+/- Range (%)", min_value=1, max_value=50, value=10, key=f"tol_{trait}")
                        active_filters[trait] = {'target': target_val, 'tolerance': tolerance}
                    st.write("") 
        submit_search = st.form_submit_button("Run Search")

    if submit_search:
        results_df = active_df.copy() # Fixed: Was previously referencing global 'df'

        for trait, settings in active_filters.items():
            target = settings['target']
            tol_percent = settings['tolerance'] / 100.0
            min_bound, max_bound = target - (target * tol_percent), target + (target * tol_percent)
            results_df = results_df[(results_df[trait] >= min_bound) & (results_df[trait] <= max_bound)]

        if use_map_filter:
            # Dynamically find the coordinate columns for this dataset
            lat_col = 'lat_centroid' if 'lat_centroid' in results_df.columns else 'Centroid.Latitude'
            lon_col = 'lon_centroid' if 'lon_centroid' in results_df.columns else 'Centroid.Longitude'
            range_col = 'range_size' if 'range_size' in results_df.columns else 'Range.Size'

            # 1. Drop true missing values
            results_df = results_df.dropna(subset=[lat_col, lon_col, range_col])
            
            # 2. THE NULL ISLAND FIREWALL (Strips out the fake 0, 0 coordinates)
            results_df = results_df[~((results_df[lat_col] == 0) & (results_df[lon_col] == 0))]

            # 3. Safely calculate the distance and overlap percentage
            overlaps = results_df.apply(lambda row: calculate_overlap_percentage(target_lat, target_lon, search_radius, row[lat_col], row[lon_col], row[range_col]), axis=1)
            results_df['Overlap_%'] = overlaps
            results_df = results_df[results_df['Overlap_%'] >= min_overlap]

        st.subheader("🎯 Match Results")
        
        if len(active_filters) == 0 and not use_map_filter:
            st.info("👈 Enable at least one filter above (Geographic or Trait) and click Run Search.")
        elif results_df.empty:
            st.warning("No birds found matching those exact parameters.")
        else:
            st.session_state['tab2_results'] = results_df
            st.success(f"Found {len(results_df)} matching species!")
            
            # --- 1. BULLETPROOF COLUMN SELECTOR ---
            # We strictly check the active results_df to see what names it has
            columns_to_show = [c for c in ['species_birdlife', 'species_birdtree'] if c in results_df.columns]
            
            # If no names are found, grab the safest available ID column
            if not columns_to_show:
                if 'avibase_id' in results_df.columns: columns_to_show.append('avibase_id')
                elif 'Avibase.ID' in results_df.columns: columns_to_show.append('Avibase.ID')
                else: columns_to_show.append(results_df.columns[0])
                
            # Add the trait filters to the table
            columns_to_show += [trait for trait in active_filters.keys() if trait in results_df.columns]
            
            if use_map_filter:
                results_df['Overlap_%'] = results_df['Overlap_%'].round(2)
                if 'Overlap_%' not in columns_to_show:
                    columns_to_show.append('Overlap_%')
                
            st.dataframe(results_df[columns_to_show])
            
            # --- 2. DOWNLOAD BUTTONS ---
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(label="📄 Download Summary", data=results_df[columns_to_show].to_csv(index=False).encode('utf-8'), file_name='bird_match_summary.csv', mime='text/csv')
            with dl_col2:
                st.download_button(label="📦 Download Full Dataset", data=results_df.to_csv(index=False).encode('utf-8'), file_name='bird_match_full.csv', mime='text/csv')

            # --- 3. RESULTS MAP ---
            st.write("---")
            st.subheader("🗺️ Geographic Distribution of Matches")
            if use_map_filter and min_overlap < 35:
                st.info("💡 Some matched birds may have centers outside your circle because their total continental range overlaps your circle.")

            fig_results = go.Figure()
            if use_map_filter:
                fig_results.add_trace(go.Scattergeo(lat=circle_lats, lon=circle_lons, mode='lines', line=dict(color='red', width=2), fill='toself', fillcolor='rgba(255, 0, 0, 0.1)', name='Your Search Region'))

            map_results_df = results_df.dropna(subset=[cols['lat'], cols['lon'], cols['range']])

            if not map_results_df.empty:
                max_range = map_results_df[cols['range']].max()
                marker_sizes = (map_results_df[cols['range']] / max_range * 40) + 6 if max_range > 0 else 8

                # Safe Hover Text logic for the map
                if 'species_birdlife' in map_results_df.columns: hover_text = map_results_df['species_birdlife']
                elif 'avibase_id' in map_results_df.columns: hover_text = map_results_df['avibase_id']
                elif 'Avibase.ID' in map_results_df.columns: hover_text = map_results_df['Avibase.ID']
                else: hover_text = map_results_df.iloc[:, 0]

                fig_results.add_trace(go.Scattergeo(
                    lat=map_results_df[cols['lat']],
                    lon=map_results_df[cols['lon']],
                    customdata=np.stack((map_results_df.get('Overlap_%', np.zeros(len(map_results_df))), map_results_df[cols['range']]), axis=-1),
                    hovertemplate="<i>%{text}</i><br>Range: %{customdata[1]:,.0f} sq km<extra></extra>",
                    text=hover_text,
                    mode='markers',
                    marker=dict(size=marker_sizes, color='#3b82f6', opacity=0.7, line=dict(width=1, color='black')),
                    name='Matched Birds'
                ))
                
            fig_results.update_geos(showcountries=True, showcoastlines=True, countrycolor="LightGrey", projection_type="natural earth")
            if use_map_filter: fig_results.update_geos(center=dict(lat=target_lat, lon=target_lon), projection_scale=2.5)
            st.plotly_chart(fig_results, use_container_width=True)       
    
# ==========================================
# TAB 3: EXPLORATORY ANALYSIS & PCA
# ==========================================
with tab3:
    st.header("📊 Exploratory Data Analysis & Distributions")
    active_df = st.session_state['working_df']
    
    # --- DYNAMIC COLUMN SCANNER ---
    # Automatically find all numeric columns (for math) and text columns (for categories)
    numeric_cols = active_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = active_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- PART 1: INTERACTIVE HISTOGRAMS ---
    if numeric_cols:
        st.write("Explore the distribution of individual traits across the dataset.")
        
        # Default to mass or wing length if available
        hist_default = 0
        if 'mass' in numeric_cols: hist_default = numeric_cols.index('mass')
        elif 'mass_avg' in numeric_cols: hist_default = numeric_cols.index('mass_avg')
            
        selected_dist = st.selectbox("Select a trait to view its distribution:", options=numeric_cols, index=hist_default)
        
        fig_hist = px.histogram(
            active_df, x=selected_dist, nbins=50, 
            title=f"Distribution of {selected_dist}",
            template="plotly_white", color_discrete_sequence=['#3b82f6']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No numerical columns found for distribution plotting.")
        
    st.write("---")

    # --- PART 2A: CATEGORICAL BAR CHARTS ---
    st.subheader("📋 Categorical Trait Distributions")
    
    if categorical_cols:
        # Default to habitat or order if available
        cat_default = 0
        if 'habitat' in categorical_cols: cat_default = categorical_cols.index('habitat')
            
        selected_cat = st.selectbox("Select a category to view the top 15 groups:", options=categorical_cols, index=cat_default)
        
        counts = active_df[selected_cat].value_counts().head(15).reset_index()
        counts.columns = [selected_cat, 'Count']
        
        fig_bar = px.bar(
            counts, x=selected_cat, y='Count', 
            title=f"Top 15 {selected_cat} Groups",
            template="plotly_white", color_discrete_sequence=['#f56565'] 
        )
        fig_bar.update_layout(xaxis_tickangle=-45) 
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- PART 2B: INTERACTIVE BOXPLOTS ---
    st.write("---")
    st.subheader("📦 Trait Variation by Category (Boxplots)")
    st.write("Compare how a numeric measurement changes across different categories.")

    if categorical_cols and numeric_cols:
        box_col1, box_col2 = st.columns(2)
        with box_col1: 
            box_cat = st.selectbox("Select X-Axis (Grouping Category):", options=categorical_cols, index=cat_default, key='box_cat')
        with box_col2: 
            box_num = st.selectbox("Select Y-Axis (Numeric Trait):", options=numeric_cols, index=hist_default, key='box_num')

        clean_box_df = active_df.dropna(subset=[box_cat, box_num])

        if not clean_box_df.empty:
            fig_box = px.box(
                clean_box_df, x=box_cat, y=box_num, color=box_cat, 
                title=f"Distribution of {box_num} across {box_cat}", template="plotly_white"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning(f"⚠️ No data available to plot **{box_num}** across **{box_cat}**. The overlapping rows only contain missing values.")

    # --- PART 3: INTERACTIVE PCA MODEL ---
    st.write("---")
    st.subheader("🧬 Interactive PCA Cluster Analysis")
    st.write("Run a custom Principal Component Analysis (PCA) to reduce dimensions and view clustering.")

    if categorical_cols and len(numeric_cols) >= 2:
        pca_col1, pca_col2 = st.columns(2)

        with pca_col1:
            # Try to default to some standard measurements
            default_pca_traits = [c for c in ['wing_len', 'mass', 'beak_culmen', 'tarsus', 'wing_len_avg', 'mass_avg'] if c in numeric_cols]
            if len(default_pca_traits) < 2: default_pca_traits = numeric_cols[:2]
                
            selected_pca_traits = st.multiselect("Select Numeric Traits for PCA:", options=numeric_cols, default=default_pca_traits)

        with pca_col2:
            pca_group = st.selectbox("Select Category to Color By (Hue):", options=categorical_cols, index=cat_default, key='pca_cat')

        if len(selected_pca_traits) >= 2:
            df_pca_ready = active_df.dropna(subset=selected_pca_traits + [pca_group]).copy()

            if len(df_pca_ready) > 5:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_pca_ready[selected_pca_traits])

                pca = PCA(n_components=2)
                pcs = pca.fit_transform(X_scaled)

                df_pca_ready['PC1'] = pcs[:, 0]
                df_pca_ready['PC2'] = pcs[:, 1]

                var_pc1 = pca.explained_variance_ratio_[0] * 100
                var_pc2 = pca.explained_variance_ratio_[1] * 100

                fig_pca = px.scatter(
                    df_pca_ready, x='PC1', y='PC2', color=pca_group,
                    hover_name='species_birdlife' if 'species_birdlife' in df_pca_ready.columns else None,
                    title=f"PCA of Selected Traits (Colored by {pca_group})",
                    labels={'PC1': f'PC1 ({var_pc1:.1f}%)', 'PC2': f'PC2 ({var_pc2:.1f}%)'},
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Alphabet 
                )
                
                fig_pca.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_pca, use_container_width=True)

                st.write("**PCA Loadings (Feature Importance):**")
                st.write("This table shows how much each physical trait contributes to the X and Y axes of the graph.")
                
                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=selected_pca_traits)
                st.dataframe(loadings.style.background_gradient(cmap='Blues')) 

            else:
                st.warning("⚠️ Not enough complete rows to run PCA on these selected traits.")
        else:
            st.info("Please select at least 2 numeric traits to run the PCA model.")
# ==========================================
# TAB 4: ADVANCED BIOGEOGRAPHICAL MAP
# ==========================================
with tab4:
    st.header("🌍 Advanced Biogeographical Range Map")
    active_df = st.session_state['working_df']
    
    # --- 1. DYNAMIC COLUMN SCANNER ---
    lat_col = 'lat_centroid' if 'lat_centroid' in active_df.columns else 'Centroid.Latitude'
    lon_col = 'lon_centroid' if 'lon_centroid' in active_df.columns else 'Centroid.Longitude'
    range_col = 'range_size' if 'range_size' in active_df.columns else 'Range.Size'
    name_col = 'species_birdlife' if 'species_birdlife' in active_df.columns else active_df.columns[0]
    
    # Check for bounding box data for the single-species viewer
    min_lat = 'lat_min' if 'lat_min' in active_df.columns else 'Min.Latitude'
    max_lat = 'lat_max' if 'lat_max' in active_df.columns else 'Max.Latitude'
    min_lon = 'lon_min' if 'lon_min' in active_df.columns else 'Min.Longitude'
    max_lon = 'lon_max' if 'lon_max' in active_df.columns else 'Max.Longitude'
    
    has_bbox = all(c in active_df.columns for c in [min_lat, max_lat, min_lon, max_lon])

    if lat_col in active_df.columns and lon_col in active_df.columns:
        st.write("Visualize how species are distributed globally.")
        
        # --- 2. THE NULL ISLAND FIREWALL ---
        # Drop true missing values (NaNs)
        map_df = active_df.dropna(subset=[lat_col, lon_col]).copy()
        
        # Filter out coordinates exactly at (0.0, 0.0) to remove the "Null Island" artifact!
        map_df = map_df[~((map_df[lat_col] == 0) & (map_df[lon_col] == 0))]

        # UI Toggle
        map_mode = st.radio("Visualization Mode:", ["🗺️ Single Species Range Explorer", "🔵 Multi-Species Centroid Clusters"], horizontal=True)

        fig_map = go.Figure()

        # ---------------------------------------------------------
        # MODE 1: SINGLE SPECIES POLYGON
        # ---------------------------------------------------------
        if map_mode == "🗺️ Single Species Range Explorer":
            if has_bbox:
                clean_box_df = map_df.dropna(subset=[min_lat, max_lat, min_lon, max_lon]).copy()
                
                if not clean_box_df.empty:
                    bird_list = clean_box_df[name_col].unique()
                    selected_map_bird = st.selectbox("Select a species to view its geographic range polygon:", options=bird_list)
                    bird_row = clean_box_df[clean_box_df[name_col] == selected_map_bird].iloc[0]

                    # 4 Corners of the box
                    box_lats = [bird_row[min_lat], bird_row[max_lat], bird_row[max_lat], bird_row[min_lat], bird_row[min_lat]]
                    box_lons = [bird_row[min_lon], bird_row[min_lon], bird_row[max_lon], bird_row[max_lon], bird_row[min_lon]]

                    fig_map.add_trace(go.Scattergeo(
                        lon=box_lons, lat=box_lats, mode='lines', fill='toself', fillcolor='rgba(59, 130, 246, 0.4)',
                        line=dict(color='#2563eb', width=2), name=f"Range Area"
                    ))

                    fig_map.add_trace(go.Scattergeo(
                        lon=[bird_row[lon_col]], lat=[bird_row[lat_col]], mode='markers',
                        marker=dict(size=12, color='red', symbol='star'), name="Population Centroid"
                    ))
                else:
                    st.warning("No species found with complete bounding box coordinates.")
            else:
                st.error("Bounding box columns (Min/Max Latitude & Longitude) not found in this dataset.")
                
        # ---------------------------------------------------------
        # MODE 2: MULTI-SPECIES DOT CLUSTERS
        # ---------------------------------------------------------
        else:
            # Handle Range Size safety
            if range_col in map_df.columns:
                map_df[range_col] = map_df[range_col].fillna(map_df[range_col].median())
            else:
                map_df['Plot_Size'] = 10 # Fallback size if range is completely missing
                range_col = 'Plot_Size'

            # Dynamically find text columns for coloring
            categorical_cols = map_df.select_dtypes(include=['object', 'category']).columns.tolist()
            color_options = [c for c in ['habitat', 'primary_diet', 'trophic_niche', 'lifestyle', 'migration'] if c in categorical_cols]
            
            map_color = None
            if color_options:
                map_color = st.selectbox("Color map points by:", options=color_options)
                map_df[map_color] = map_df[map_color].fillna("Unknown").astype(str)

            # Draw the px map and port it to go.Figure
            fig_px = px.scatter_geo(
                map_df, lat=lat_col, lon=lon_col, hover_name=name_col, size=range_col, color=map_color,
            )
            
            for trace in fig_px.data:
                fig_map.add_trace(trace)
                
            fig_map.update_traces(marker=dict(sizemin=2, sizeref=20 if range_col != 'Plot_Size' else 1))

        # ---------------------------------------------------------
        # FINAL MAP STYLING
        # ---------------------------------------------------------
        fig_map.update_layout(
            template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
            geo=dict(showcoastlines=True, showcountries=True, countrycolor="LightGrey", projection_type="natural earth")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    else:
        st.error("This dataset is missing Latitude and Longitude columns needed to draw the map.")