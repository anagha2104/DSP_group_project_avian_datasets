import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import math
import os
import plotly.graph_objects as go

# App configuration
st.set_page_config(page_title="Avian Explorer", layout="wide")

# ==========================================
# 1. DATA LOADING & STATE MANAGEMENT
# ==========================================
@st.cache_data
def load_clean_data():
    return pd.read_csv('data/raw/avonet_FE_01.csv', encoding='utf-8-sig') 

@st.cache_data
def load_merged_data():
    return pd.read_csv('data/raw/merged.csv', encoding='utf-8-sig')

@st.cache_data
def load_image_data():
    return pd.read_csv('data/raw/bird_image_links.csv', encoding='utf-8-sig')

df_clean = load_clean_data()
df_merged = load_merged_data()

# Ensure image_df is loaded safely
try:
    image_df = load_image_data()
except:
    image_df = pd.DataFrame(columns=['avibase_id', 'scientific_name', 'image_url'])

# Initialize session state for filtering and persistence
if 'tab2_results' not in st.session_state:
    st.session_state['tab2_results'] = None
if 'working_df' not in st.session_state:
    st.session_state['working_df'] = df_clean

st.sidebar.header("📁 Data Source Manager")
data_source = st.sidebar.radio(
    "Active Dataset:",
    [
        "🧪 Cleaned & Engineered Data", 
        "🤝 Merged Team Dataset",
        "🔍 Tab 2 Filtered Results", 
        "📂 Upload Custom CSV"
    ]
)

# Load the base dataframe
if data_source == "🧪 Cleaned & Engineered Data":
    base_df = df_clean
elif data_source == "🤝 Merged Team Dataset":
    base_df = df_merged
elif data_source == "🔍 Tab 2 Filtered Results":
    if st.session_state['tab2_results'] is not None:
        base_df = st.session_state['tab2_results']
    else:
        st.sidebar.warning("No filtered results found. Run a search in Tab 2 first.")
        base_df = df_clean
elif data_source == "📂 Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload .csv", type=['csv'])
    if uploaded_file is not None:
        try:
            base_df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully.")
        except:
            st.sidebar.error("Failed to parse CSV.")
            base_df = df_clean
    else:
        base_df = df_clean

# Assign active data directly (Removed Performance Sampling)
active_df = base_df.copy()
st.session_state['working_df'] = active_df
st.session_state['dataset_id'] = f"ds_{len(active_df.columns)}"

st.title("🦅 Avian Trait Database Explorer")

# ==========================================
# 2. DYNAMIC SCHEMA MAPPING (BULLETPROOF)
# ==========================================
def get_col(candidates, df):
    """Safely returns the first matching column name from a list of possibilities."""
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[-1] # Absolute fallback to prevent hard crashes

cols = {
    'id': get_col(['avibase_id', 'Avibase.ID', active_df.columns[0]], active_df),
    'name': get_col(['English Name (BirdLife > IOC > Clements>AviList)', 'species_birdlife', 'scientific_name'], active_df),
    'sci_name': get_col(['scientific_name', 'species_birdlife', 'species_birdtree', 'Species'], active_df),
    'mass': get_col(['Average Mass', 'mass', 'mass_avg', 'Mass'], active_df),
    'wing': get_col(['Wing.Length', 'wing_len', 'wing_len_avg'], active_df),
    'beak': get_col(['Beak.Length_Culmen', 'beak_culmen', 'beak_culmen_avg'], active_df),
    'tarsus': get_col(['Tarsus.Length', 'tarsus', 'tarsus_avg'], active_df),
    'lat': get_col(['Centroid.Latitude', 'lat_centroid', 'lat', 'Min.Latitude'], active_df),
    'lon': get_col(['Centroid.Longitude', 'lon_centroid', 'lon', 'Min.Longitude'], active_df),
    'range': get_col(['Range.Size', 'range_size', 'RangeSize'], active_df)
}

# ==========================================
# 3. GEOSPATIAL & OVERLAP MATH
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
    
    r_bird = np.sqrt(area_bird / np.pi) if area_bird > 0 else 0
    
    if d >= (r_user + r_bird): return 0.0 
    elif d <= abs(r_user - r_bird): return (np.pi * min(r_user, r_bird)**2 / area_bird) * 100
    else:
        part1 = r_user**2 * np.arccos((d**2 + r_user**2 - r_bird**2) / (2 * d * r_user))
        part2 = r_bird**2 * np.arccos((d**2 + r_bird**2 - r_user**2) / (2 * d * r_bird))
        part3 = 0.5 * np.sqrt(abs((-d + r_user + r_bird) * (d - r_user + r_bird) * (d + r_user - r_bird) * (d + r_user + r_bird)))
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Trait Lookup", 
    "🎯 Reverse Matcher", 
    "🧬 PCA & EDA", 
    "🌍 Global Map", 
    "🌳 Taxonomy"
])

# ==========================================
# TAB 1: INDIVIDUAL SPECIES LOOKUP
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

    id_col = cols['id']
    name_col = cols['name']
    
    safe_lib1 = active_df[name_col].fillna("Unknown") if name_col in active_df.columns else active_df[id_col]
    id_to_lib1 = dict(zip(active_df[id_col], safe_lib1))
    bird_ids = active_df[id_col].unique().tolist()

    if st.session_state.get('current_bird_id') not in bird_ids:
        if bird_ids: st.session_state['current_bird_id'] = bird_ids[0]
        else:
            st.warning("No birds match current filters. Adjust parameters in Tab 2.")
            st.stop()

    def update_selection():
        st.session_state['current_bird_id'] = st.session_state['search_selector']

    st.selectbox("Search for a Species:", options=bird_ids, format_func=lambda x: str(id_to_lib1.get(x)), key='search_selector', on_change=update_selection)

    selected_id = st.session_state['current_bird_id']
    bird_data = active_df[active_df[id_col] == selected_id]
    
    profile_col1, profile_col2 = st.columns([1, 2])
    
    with profile_col1:
        display_name = id_to_lib1.get(selected_id, "Unknown Species")
        
        # --- UNIVERSAL NAME CROSS-REFERENCER ---
        # 1. Establish the "Bridge" (The Scientific Name)
        sci_name = bird_data[cols['sci_name']].iloc[0] if cols['sci_name'] in bird_data.columns and not bird_data.empty else display_name
        
        # 2. Try to pull names directly from the active dataset
        common_name = bird_data['English Name (BirdLife > IOC > Clements>AviList)'].iloc[0] if 'English Name (BirdLife > IOC > Clements>AviList)' in bird_data.columns else None
        bl_name = bird_data['species_birdlife'].iloc[0] if 'species_birdlife' in bird_data.columns else None
        bt_name = bird_data['species_birdtree'].iloc[0] if 'species_birdtree' in bird_data.columns else None

        # 3. If names are missing, silently fetch them from the OTHER datasets using case-insensitive matching
        if pd.isna(common_name) or not common_name:
            match = df_merged[df_merged['scientific_name'].str.lower() == str(sci_name).lower()]
            if not match.empty:
                common_name = match['English Name (BirdLife > IOC > Clements>AviList)'].iloc[0]

        if pd.isna(bl_name) or not bl_name:
            match = df_clean[(df_clean['species_birdlife'].str.lower() == str(sci_name).lower()) | (df_clean['species_birdtree'].str.lower() == str(sci_name).lower())]
            if not match.empty:
                bl_name = match['species_birdlife'].iloc[0]
                bt_name = match['species_birdtree'].iloc[0]

        # --- IMAGE MATCHING (Rerouted through Scientific Name) ---
        if 'avibase_id' in image_df.columns and 'avibase_id' in bird_data.columns and pd.notna(bird_data['avibase_id'].iloc[0]):
            matching_images = image_df[image_df['avibase_id'] == bird_data['avibase_id'].iloc[0]]
        else:
            # Force the search to use scientific/BirdLife names, NEVER the common name
            img_search_name = bl_name if pd.notna(bl_name) else sci_name
            matching_images = image_df[image_df['scientific_name'].str.lower() == str(img_search_name).lower()]
            
        # Display the best available name as the caption
        best_display_name = common_name if pd.notna(common_name) else display_name
        
        if not matching_images.empty:
            st.image(matching_images['image_url'].iloc[0], caption=best_display_name, use_container_width=True)
        else:
            st.info("📷 Image not found in repository.")
            
    with profile_col2:
        st.subheader(f"Metadata: {best_display_name}")
        
        # --- UNIFIED TAXONOMY DISPLAY ---
        st.markdown("**Known Aliases & Taxonomy:**")
        if pd.notna(common_name):
            st.markdown(f"- **Common Name:** {common_name}")
        if pd.notna(sci_name):
            st.markdown(f"- **Scientific Name:** {sci_name}")
        if pd.notna(bl_name):
            st.markdown(f"- **BirdLife Database Name:** {bl_name}")
        if pd.notna(bt_name) and bt_name != bl_name:
            st.markdown(f"- **BirdTree Database Name:** {bt_name}")
        
        st.markdown("---")
        st.markdown("**Key Traits:**")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            if cols['mass'] in active_df.columns and not bird_data.empty:
                val = bird_data[cols['mass']].iloc[0]
                if pd.notna(val): st.metric("Mass (g)", round(val, 2)) 
        with m_col2:
            if cols['wing'] in active_df.columns and not bird_data.empty:
                val = bird_data[cols['wing']].iloc[0]
                if pd.notna(val): st.metric("Wing (mm)", round(val, 2))
        with m_col3:
            if cols['beak'] in active_df.columns and not bird_data.empty:
                val = bird_data[cols['beak']].iloc[0]
                if pd.notna(val): st.metric("Beak (mm)", round(val, 2))
                
        st.write("")
        st.dataframe(bird_data)

# ==========================================
# TAB 2: MULTI-PARAM REVERSE SEARCH
# ==========================================
with tab2:
    st.header("🎯 Target Trait Search")
    use_map_filter = st.checkbox("Apply Geographic Constraints")
    
    if use_map_filter:
        c_map1, c_map2 = st.columns([1, 2])
        with c_map1:
            target_lat = st.slider("Latitude", -90.0, 90.0, 20.0)
            target_lon = st.slider("Longitude", -180.0, 180.0, 75.0)
            search_radius = st.slider("Radius (km)", 100, 5000, 1000)
            min_overlap = st.slider("Min % Overlap", 1, 100, 50)
            
        with c_map2:
            lats, lons = generate_map_circle(target_lat, target_lon, search_radius)
            fig_pre = go.Figure(go.Scattergeo(lat=lats, lon=lons, mode='lines', fill='toself', fillcolor='rgba(255, 0, 0, 0.2)'))
            fig_pre.update_geos(center=dict(lat=target_lat, lon=target_lon), projection_scale=3)
            fig_pre.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_pre, use_container_width=True)

    searchable = [cols['mass'], cols['wing'], cols['beak'], cols['tarsus']]
    active_filters = {}

    with st.form("search_form"):
        for trait in searchable:
            if trait in active_df.columns:
                if st.checkbox(f"Filter by {trait}"):
                    fc1, fc2 = st.columns(2)
                    with fc1: t_val = st.number_input(f"Target {trait}", value=50.0, key=f"t_{trait}")
                    with fc2: tol = st.slider(f"Tolerance %", 1, 50, 10, key=f"tol_{trait}")
                    active_filters[trait] = {'target': t_val, 'tol': tol}
        run = st.form_submit_button("Search Database")

    if run:
        results = active_df.copy()
        for trait, s in active_filters.items():
            b = s['target'] * (s['tol'] / 100.0)
            results = results[(results[trait] >= s['target'] - b) & (results[trait] <= s['target'] + b)]

        if use_map_filter and cols['lat'] in results.columns and cols['lon'] in results.columns:
            results = results.dropna(subset=[cols['lat'], cols['lon'], cols['range']])
            results = results[~((results[cols['lat']] == 0) & (results[cols['lon']] == 0))]
            results['Overlap_%'] = results.apply(lambda r: calculate_overlap_percentage(target_lat, target_lon, search_radius, r[cols['lat']], r[cols['lon']], r[cols['range']]), axis=1)
            results = results[results['Overlap_%'] >= min_overlap]

        st.session_state['tab2_results'] = results
        st.success(f"Matches found: {len(results)}")
        st.dataframe(results)

# ==========================================
# TAB 3: ANALYSIS (HISTOGRAMS & PCA)
# ==========================================
with tab3:
    st.header("📊 Trait Analysis & Dimension Reduction")
    fp = str(list(active_df.columns[:5]))
    ds_id = st.session_state['dataset_id']

    exclude = ['id', 'species', 'name', 'source', 'inference', 'count', 'total', 'order', 'family']
    
    numeric_cols = [c for c in active_df.select_dtypes(include=np.number).columns if not any(k in c.lower() for k in exclude)]
    categorical_cols = [c for c in active_df.select_dtypes(include=['object', 'category']).columns if not any(k in c.lower() for k in exclude) and active_df[c].nunique() < 100]

    if numeric_cols:
        trait_dist = st.selectbox("Distribution Trait:", numeric_cols, key=f"h_{fp}")
        st.plotly_chart(px.histogram(active_df, x=trait_dist, template="plotly_white"), use_container_width=True)

    st.write("---")
    
    if categorical_cols and numeric_cols:
        st.subheader("Boxplot Comparisons")
        c1, c2 = st.columns(2)
        with c1: b_cat = st.selectbox("Category:", categorical_cols, key=f"bc_{ds_id}")
        with c2: b_num = st.selectbox("Metric:", numeric_cols, key=f"bn_{ds_id}")
        st.plotly_chart(px.box(active_df.dropna(subset=[b_cat, b_num]), x=b_cat, y=b_num, color=b_cat), use_container_width=True)

    # ==========================================
    # NEW: TRAIT VS. TRAIT SCATTER PLOT
    # ==========================================
    st.write("---")
    st.subheader("📈 Trait Comparison (Scatter Plot)")
    
    # We need at least 2 numeric columns to make a scatter plot
    if len(numeric_cols) >= 2:
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        
        with sc_col1:
            scat_x = st.selectbox("X-Axis (Numeric):", numeric_cols, index=0, key=f"sx_{ds_id}")
        with sc_col2:
            scat_y = st.selectbox("Y-Axis (Numeric):", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key=f"sy_{ds_id}")
        with sc_col3:
            scat_color = st.selectbox("Color By (Optional):", [None] + categorical_cols, key=f"scolor_{ds_id}")

        # Filter out missing data just for the selected columns
        subset_cols = [scat_x, scat_y] + ([scat_color] if scat_color else [])
        scat_df = active_df.dropna(subset=subset_cols).copy()

        if not scat_df.empty:
            # We use the dynamic name column for the hover text so you know which bird is which dot!
            hover_name = cols['name'] if cols['name'] in scat_df.columns else None
            
            fig_scat = px.scatter(
                scat_df, 
                x=scat_x, 
                y=scat_y, 
                color=scat_color,
                hover_name=hover_name,
                template="plotly_white",
                opacity=0.7, # Replicating your alpha=0.7 from seaborn
                title=f"{scat_y} vs {scat_x}"
            )
            
            # Making it look clean
            fig_scat.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.warning("⚠️ No valid data to plot for these specific selections.")


    st.write("---")
    st.subheader("🧬 Principal Component Analysis")
    
    if len(numeric_cols) >= 2:
        pca_traits = st.multiselect("PCA Features:", numeric_cols, key=f"pm_{ds_id}")
        if categorical_cols:
            pca_color = st.selectbox("Color By:", categorical_cols, key=f"pc_{ds_id}")
        else:
            pca_color = None

        if len(pca_traits) >= 2:
            subset_cols = pca_traits + ([pca_color] if pca_color else [])
            pca_data = active_df.dropna(subset=subset_cols).copy()
            
            if len(pca_data) > 5:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(pca_data[pca_traits])
                
                pca = PCA(n_components=2)
                coords = pca.fit_transform(scaled)
                pca_data['PC1'], pca_data['PC2'] = coords[:, 0], coords[:, 1]
                
                fig_pca = px.scatter(pca_data, x='PC1', y='PC2', color=pca_color, template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)
                
                st.write("Feature Contributions:")
                st.dataframe(pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_traits))
            else:
                st.warning("Not enough clean data to run PCA.")

# ==========================================
# TAB 4: 3D SPATIAL GLOBAL VIEW
# ==========================================
with tab4:
    st.header("🌍 Global Distribution")
    
    if cols['lat'] in active_df.columns and cols['lon'] in active_df.columns:
        with st.spinner("Rendering geographic data..."):
            map_df = active_df.dropna(subset=[cols['lat'], cols['lon']]).copy()
            map_df = map_df[~((map_df[cols['lat']] == 0) & (map_df[cols['lon']] == 0))]

            if not map_df.empty:
                map_df['Size'] = map_df[cols['range']].fillna(map_df[cols['range']].median()) if cols['range'] in map_df.columns else 10
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    m_color = st.selectbox("Color by:", categorical_cols, key="map_c") if categorical_cols else None
                with m_col2:
                    m_scale = st.slider("Marker Scale:", 0.1, 3.0, 0.5)

                fig_globe = px.scatter_geo(map_df, lat=cols['lat'], lon=cols['lon'], color=m_color, size='Size', projection="orthographic", template="plotly_white")
                fig_globe.update_geos(showocean=True, oceancolor="#e0f2fe", showland=True, landcolor="White")
                st.plotly_chart(fig_globe, use_container_width=True)
            else:
                st.warning("No valid geographic coordinates found.")
    else:
        st.warning("Geographic data (Latitude/Longitude) is missing from this dataset.")

# ==========================================
# TAB 5: TAXONOMIC HIERARCHY
# ==========================================
with tab5:
    st.header("🌳 Taxonomy Explorer")
    
    order_col = get_col(['order_birdlife', 'Order_x', 'order_birdtree', 'Order'], active_df)
    family_col = get_col(['family_birdlife', 'Family IOC 15.1', 'family_birdtree', 'Family'], active_df)
    ds_id = st.session_state.get('dataset_id', 'default')

    if order_col in active_df.columns and family_col in active_df.columns:
        numeric_params = active_df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            color_trait = st.selectbox("Color by (Mean Value):", [None] + numeric_params, key=f"sun_c_{ds_id}")
        with col2:
            st.info("⚡ Optimization: Chart is pre-aggregated by Family for speed.")

        agg_dict = {active_df.columns[0]: 'count'}
        if color_trait: agg_dict[color_trait] = 'mean'
            
        sunburst_df = active_df.groupby([order_col, family_col]).agg(agg_dict).reset_index()
        sunburst_df.columns = [order_col, family_col, 'Count'] + ([color_trait] if color_trait else [])

        fig_sun = px.sunburst(
            sunburst_df,
            path=[order_col, family_col], 
            values='Count',
            color=color_trait if color_trait else order_col,
            color_continuous_scale='Viridis',
            template="plotly_white"
        )
        fig_sun.update_layout(height=700)
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.warning("Taxonomic columns (Order/Family) not found in this dataset.")