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
    # Use utf-8-sig to handle BOM and prevent KeyErrors on first column
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

# Initialize session state for filtering and persistence
if 'tab2_results' not in st.session_state:
    st.session_state['tab2_results'] = None
if 'working_df' not in st.session_state:
    st.session_state['working_df'] = df_clean

# Generate a unique ID based on column count to force widget refresh on dataset swap
st.session_state['dataset_id'] = f"ds_{len(st.session_state['working_df'].columns)}"

st.sidebar.header(" Data Source Manager")
data_source = st.sidebar.radio(
    "Active Dataset:",
    [
        " Cleaned & Engineered Data", 
        " Raw Data (With Missing Values)", 
        " Tab 2 Filtered Results", 
        " Upload Custom CSV"
    ]
)

if data_source == " Cleaned & Engineered Data":
    st.session_state['working_df'] = df_clean
elif data_source == " Raw Data (With Missing Values)":
    st.session_state['working_df'] = df_raw
    st.sidebar.info(" PCA and Histograms may exclude rows containing NaNs.")
elif data_source == " Tab 2 Filtered Results":
    if st.session_state['tab2_results'] is not None:
        st.session_state['working_df'] = st.session_state['tab2_results']
    else:
        st.sidebar.warning("No filtered results found. Run a search in Tab 2 first.")
        st.session_state['working_df'] = df_clean
elif data_source == " Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload .csv", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state['working_df'] = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully.")
        except:
            st.sidebar.error("Failed to parse CSV.")

st.title(" Avian Trait Database Explorer")

# ==========================================
# 2. DYNAMIC SCHEMA MAPPING
# ==========================================
# Handles inconsistent naming conventions across different AVONET versions
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
# 3. GEOSPATIAL & OVERLAP MATH
# ==========================================
def calculate_overlap_percentage(lat1, lon1, r_user, lat2, lon2, area_bird):
    """ Calculates how much of a bird's range falls within the user's circular search area. """
    R_earth = 6371.0 
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    # Haversine distance
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R_earth * c
    
    r_bird = np.sqrt(area_bird / np.pi)
    
    if d >= (r_user + r_bird): return 0.0 
    elif d <= abs(r_user - r_bird): return (np.pi * min(r_user, r_bird)**2 / area_bird) * 100
    else:
        # Circle-Circle Intersection Area
        part1 = r_user**2 * np.arccos((d**2 + r_user**2 - r_bird**2) / (2 * d * r_user))
        part2 = r_bird**2 * np.arccos((d**2 + r_bird**2 - r_user**2) / (2 * d * r_bird))
        part3 = 0.5 * np.sqrt((-d + r_user + r_bird) * (d - r_user + r_bird) * (d + r_user - r_bird) * (d + r_user + r_bird))
        return ((part1 + part2 - part3) / area_bird) * 100

def generate_map_circle(lat, lon, radius_km, num_points=64):
    """ Generates coordinates for a circle on a sphere using destination point math. """
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

# Layout initialization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Trait Lookup", 
    " Reverse Matcher", 
    " PCA & EDA", 
    " Global Map", 
    " Taxonomy"
])

# ==========================================
# TAB 1: INDIVIDUAL SPECIES LOOKUP
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

    # Dynamic ID identification
    id_col = cols['id']

    # Fast lookup mapping for UI selectors
    safe_lib1 = active_df['species_birdlife'].fillna("Unknown") if 'species_birdlife' in active_df.columns else active_df[id_col]
    safe_lib2 = active_df['species_birdtree'].fillna("Unknown") if 'species_birdtree' in active_df.columns else active_df[id_col]
    
    id_to_lib1 = dict(zip(active_df[id_col], safe_lib1))
    id_to_lib2 = dict(zip(active_df[id_col], safe_lib2))
    bird_ids = active_df[id_col].unique().tolist()

    # Reset current selection if filters exclude the previously selected species
    if st.session_state.get('current_bird_id') not in bird_ids:
        if bird_ids: st.session_state['current_bird_id'] = bird_ids[0]
        else:
            st.warning("No birds match current filters. Adjust parameters in Tab 2.")
            st.stop()

    def update_from_lib1():
        st.session_state['current_bird_id'] = st.session_state['lib1_selector']
        st.session_state['lib2_selector'] = st.session_state['lib1_selector']
        
    def update_from_lib2():
        st.session_state['current_bird_id'] = st.session_state['lib2_selector']
        st.session_state['lib1_selector'] = st.session_state['lib2_selector']

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.selectbox("BirdLife Name:", options=bird_ids, format_func=lambda x: str(id_to_lib1.get(x)), key='lib1_selector', on_change=update_from_lib1)
    with search_col2:
        st.selectbox("BirdTree Name:", options=bird_ids, format_func=lambda x: str(id_to_lib2.get(x)), key='lib2_selector', on_change=update_from_lib2)

    selected_id = st.session_state['current_bird_id']
    bird_data = active_df[active_df[id_col] == selected_id]
    
    profile_col1, profile_col2 = st.columns([1, 2])
    
    with profile_col1:
        display_name = id_to_lib1.get(selected_id, "Unknown Species")
        matching_images = image_df[image_df['scientific_name'] == display_name]
            
        if not matching_images.empty:
            st.image(matching_images['image_url'].iloc[0], caption=display_name, use_container_width=True)
        else:
            st.info(" Image not found in repository.")
            
    with profile_col2:
        st.subheader(f"Metadata: {display_name}")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            if cols['mass'] in active_df.columns:
                st.metric("Mass (g)", round(bird_data[cols['mass']].iloc[0], 2)) 
        with m_col2:
            if cols['wing'] in active_df.columns:
                st.metric("Wing (mm)", round(bird_data[cols['wing']].iloc[0], 2))
        with m_col3:
            if cols['beak'] in active_df.columns:
                st.metric("Beak (mm)", round(bird_data[cols['beak']].iloc[0], 2))
                
        st.dataframe(bird_data)


# ==========================================
# TAB 2: MULTI-PARAM REVERSE SEARCH
# ==========================================
with tab2:
    st.header(" Target Trait Search")
    
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

    # Dynamic filter generation based on available traits
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

        if use_map_filter:
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
    st.header(" Trait Analysis & Dimension Reduction")
    
    active_df = st.session_state['working_df']
    fp = str(list(active_df.columns[:5]))
    ds_id = st.session_state['dataset_id']

    # Filter out non-trait columns for cleaner UI
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

    st.write("---")
    st.subheader(" Principal Component Analysis")
    
    if len(numeric_cols) >= 2:
        pca_traits = st.multiselect("PCA Features:", numeric_cols, key=f"pm_{ds_id}")
        pca_color = st.selectbox("Color By:", categorical_cols, key=f"pc_{ds_id}")

        if len(pca_traits) >= 2:
            pca_data = active_df.dropna(subset=pca_traits + [pca_color]).copy()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(pca_data[pca_traits])
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaled)
            pca_data['PC1'], pca_data['PC2'] = coords[:, 0], coords[:, 1]
            
            fig_pca = px.scatter(pca_data, x='PC1', y='PC2', color=pca_color, template="plotly_white")
            st.plotly_chart(fig_pca, use_container_width=True)
            
            st.write("Feature Contributions:")
            st.dataframe(pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_traits))

# ==========================================
# TAB 4: 3D SPATIAL GLOBAL VIEW
# ==========================================
with tab4:
    st.header(" Global Distribution")
    
    map_df = active_df.dropna(subset=[cols['lat'], cols['lon']]).copy()
    map_df = map_df[~((map_df[cols['lat']] == 0) & (map_df[cols['lon']] == 0))]

    if not map_df.empty:
        # Prevent markers from being too small or huge
        map_df['Size'] = map_df[cols['range']].fillna(map_df[cols['range']].median()) if cols['range'] in map_df.columns else 10
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            m_color = st.selectbox("Color by:", categorical_cols, key="map_c")
        with m_col2:
            m_scale = st.slider("Marker Scale:", 0.1, 3.0, 0.5)

        fig_globe = px.scatter_geo(map_df, lat=cols['lat'], lon=cols['lon'], color=m_color, size='Size', projection="orthographic", template="plotly_white")
        fig_globe.update_geos(showocean=True, oceancolor="#e0f2fe", showland=True, landcolor="White")
        st.plotly_chart(fig_globe, use_container_width=True)

# ==========================================
# TAB 5: TAXONOMIC HIERARCHY
# ==========================================
with tab5:
    st.header(" Taxonomy Explorer")
    
    order_col = 'order_birdlife' if 'order_birdlife' in active_df.columns else 'order_birdtree'
    family_col = 'family_birdlife' if 'family_birdlife' in active_df.columns else 'family_birdtree'
    
    if order_col in active_df.columns:
        tax_color = st.selectbox("Color Hierarchy By:", [None] + numeric_cols, key=f"sc_{ds_id}")
        
        # Plotly Sunburst rendering
        fig_sun = px.sunburst(
            active_df.dropna(subset=[order_col, family_col]),
            path=[order_col, family_col], 
            color=tax_color if tax_color else order_col,
            template="plotly_white"
        )
        fig_sun.update_layout(height=700)
        st.plotly_chart(fig_sun, use_container_width=True)