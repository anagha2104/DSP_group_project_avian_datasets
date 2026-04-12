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
    return pd.read_csv('data_used/raw/avonet_FE_01.csv', encoding='utf-8-sig') 

@st.cache_data
def load_raw_data():
    return pd.read_csv('data_used/raw/avonet_cleaned.csv', encoding='utf-8-sig')

@st.cache_data
def load_image_data():
    return pd.read_csv('data_used/raw/bird_image_links.csv', encoding='utf-8-sig')

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

# Layout initialization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Trait Lookup", "🎯 Reverse Matcher", "🧬 Trait Relationship Analysis and PCA", "🌍 Global Map", "🌳 Taxonomic Diversity Explore"])

# ==========================================
# TAB 1: INDIVIDUAL SPECIES LOOKUP
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

   # Dynamic ID identification
    if 'avibase_id' in active_df.columns:
        id_col = 'avibase_id'
    elif 'Avibase.ID' in active_df.columns:
        id_col = 'Avibase.ID'
    else:
        id_col = active_df.columns[0] # Ultimate fallback

    # Fast lookup mapping for UI selectors
    safe_lib1 = active_df['species_birdlife'].fillna("Name Unavailable") if 'species_birdlife' in active_df.columns else active_df[cols['id']]
    safe_lib2 = active_df['species_birdtree'].fillna("Name Unavailable") if 'species_birdtree' in active_df.columns else active_df[cols['id']]
    
    id_to_lib1 = dict(zip(active_df[cols['id']], safe_lib1))
    id_to_lib2 = dict(zip(active_df[cols['id']], safe_lib2))
    bird_ids = active_df[cols['id']].unique().tolist()

    # --- THE PHANTOM BIRD FIX ---
    # If the user filtered the dataset and the currently saved bird is no longer in it,
    # we MUST reset the memory to the first available bird in the new filtered list!
    # Reset current selection if filters exclude the previously selected species
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
# TAB 2: MULTI-PARAMETER REVERSE SEARCH
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

    # Dynamic filter generation based on available traits
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
# TAB 3: ANALYSIS (HISTOGRAMS & PCA)
# ==========================================
with tab3:
    st.header("📊 Exploratory Data Analysis & Distributions")
    
    # --- 1. THE DATASET RECOGNIZER ---
    # We create a unique fingerprint based on the columns in the active dataframe
    active_df = st.session_state['working_df']
    dataset_fingerprint = str(list(active_df.columns[:5])) # Unique ID based on headers
    
    # --- 1. THE "CLEAN" COLUMN SCANNER ---
    # List of keywords to exclude from our analysis dropdowns
    exclude_keywords = ['id', 'species', 'name', 'source', 'inference', 'count', 'total', 'order', 'family']

    # Filter Numeric Cols: Must be numbers AND not contain an excluded keyword
    numeric_cols = [
        col for col in active_df.select_dtypes(include=np.number).columns 
        if not any(key in col.lower() for key in exclude_keywords)
    ]

    # Filter Categorical Cols: Must be text AND not contain an excluded keyword
    # We also exclude columns that have too many unique values (like 11,000 unique species names)
    categorical_cols = [
        col for col in active_df.select_dtypes(include=['object', 'category']).columns 
        if not any(key in col.lower() for key in exclude_keywords) 
        and active_df[col].nunique() < 100  # Only show categories with less than 100 groups (like Habitat/Order)
    ]

    # --- 3. THE FIREWALL ---
    # If the user switches datasets, we ensure we don't use old, dead column names
    if numeric_cols:
        # Re-verify that our defaults actually exist in THIS specific dataset
        hist_idx = 0
        for preferred in ['mass', 'mass_avg', 'beak_culmen']:
            if preferred in numeric_cols:
                hist_idx = numeric_cols.index(preferred)
                break
        
        # We add the fingerprint to the key to FORCE a fresh widget on dataset swap
        selected_dist = st.selectbox(
            "Select a trait to view its distribution:", 
            options=numeric_cols, 
            index=hist_idx,
            key=f"hist_select_{dataset_fingerprint}" 
        )
        
        fig_hist = px.histogram(
            active_df, x=selected_dist, nbins=50, 
            title=f"Distribution of {selected_dist}",
            template="plotly_white", color_discrete_sequence=['#3b82f6']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.write("---")

    # --- PART 2B: BOXPLOTS (Applying the same key logic) ---
    if categorical_cols and numeric_cols:
        st.subheader("📦 Trait Variation by Category")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            box_cat = st.selectbox("Category:", options=categorical_cols, key=f"bc_{st.session_state['dataset_id']}")
        with bcol2:
            box_num = st.selectbox("Numeric:", options=numeric_cols, key=f"bn_{st.session_state['dataset_id']}")

        # Final check before plotting
        if box_cat in active_df.columns and box_num in active_df.columns:
            clean_box_df = active_df.dropna(subset=[box_cat, box_num])
            if not clean_box_df.empty:
                fig_box = px.box(clean_box_df, x=box_cat, y=box_num, color=box_cat)
                st.plotly_chart(fig_box, use_container_width=True)

    # --- PART 3: INTERACTIVE PCA MODEL ---
    st.write("---")
    st.subheader("🧬 Interactive PCA Cluster Analysis")
    st.write("Run a custom Principal Component Analysis (PCA) to reduce dimensions and view clustering.")

    if categorical_cols and len(numeric_cols) >= 2:
        pca_col1, pca_col2 = st.columns(2)

        with pca_col1:
            selected_pca_traits = st.multiselect("Select Traits:", options=numeric_cols, key=f"pca_m_{st.session_state['dataset_id']}")
        with pca_col2:
            pca_group = st.selectbox("Color By:", options=categorical_cols, key=f"pca_c_{st.session_state['dataset_id']}")

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
# TAB 4: 3D BIOGEOGRAPHICAL GLOBE
# ==========================================
with tab4:
    st.header("🌍 3D Biogeographical Range Globe")
    active_df = st.session_state['working_df']
    
    # --- 1. DYNAMIC COLUMN SCANNER ---
    lat_col = 'lat_centroid' if 'lat_centroid' in active_df.columns else 'Centroid.Latitude'
    lon_col = 'lon_centroid' if 'lon_centroid' in active_df.columns else 'Centroid.Longitude'
    range_col = 'range_size' if 'range_size' in active_df.columns else 'Range.Size'
    name_col = 'species_birdlife' if 'species_birdlife' in active_df.columns else active_df.columns[0]
    
    if lat_col in active_df.columns and lon_col in active_df.columns:
        st.write("Visualize how species from the active dataset are distributed globally.")
        
        # --- 2. THE NULL ISLAND FIREWALL ---
        map_df = active_df.dropna(subset=[lat_col, lon_col]).copy()
        map_df = map_df[~((map_df[lat_col] == 0) & (map_df[lon_col] == 0))]

        if not map_df.empty:
            # Handle Range Size safety
            if range_col in map_df.columns:
                map_df[range_col] = map_df[range_col].fillna(map_df[range_col].median())
            else:
                map_df['Plot_Size'] = 10 
                range_col = 'Plot_Size'

            # UI Controls
            map_ui_col1, map_ui_col2 = st.columns(2)
            with map_ui_col1:
                categorical_cols = map_df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_options = [c for c in ['habitat', 'primary_diet', 'trophic_niche', 'lifestyle', 'migration'] if c in categorical_cols]
                
                map_color = None
                if color_options:
                    map_color = st.selectbox("Color globe points by:", options=color_options)
                    map_df[map_color] = map_df[map_color].fillna("Unknown").astype(str)
            
            with map_ui_col2:
                # Let the user control the blob size!
                size_modifier = st.slider("Adjust Marker Size:", min_value=0.1, max_value=3.0, value=0.5, step=0.1)

            # --- 3. THE 3D GLOBE ---
            fig_globe = px.scatter_geo(
                map_df, lat=lat_col, lon=lon_col, hover_name=name_col, 
                size=range_col, color=map_color, projection="orthographic", 
                template="plotly_white", title="Global Species Distribution"
            )
            
            # --- THE FIX: Mathematical Sizing ---
            if range_col != 'Plot_Size':
                max_range = map_df[range_col].max()
                # Plotly's formula for scaling areas perfectly. Base max size is 40 pixels.
                base_max_size = 40 * size_modifier
                dynamic_sizeref = 2. * max_range / (base_max_size ** 2) if base_max_size > 0 else 1
            else:
                dynamic_sizeref = 1 / size_modifier

            # Apply the sizemode='area' to prevent giant linear blobs
            fig_globe.update_traces(marker=dict(sizemin=1, sizeref=dynamic_sizeref, sizemode='area', opacity=0.7))

            # Styling
            fig_globe.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                geo=dict(
                    showcoastlines=True, showcountries=True, countrycolor="LightGrey",
                    showocean=True, oceancolor="#e0f2fe", showland=True, landcolor="White"
                )
            )
            st.plotly_chart(fig_globe, use_container_width=True)
        else:
            st.warning("No valid geographic coordinates found in this dataset after filtering out missing data.")
    else:
        st.error("This dataset is missing Latitude and Longitude columns needed to draw the globe.")

# ==========================================
# TAB 5: TAXONOMIC HIERARCHY (SUNBURST)
# ==========================================
with tab5:
    st.header("🌳 Taxonomic Diversity Explorer")
    active_df = st.session_state['working_df']
    
    # 1. Identify Taxonomy Columns
    order_col = 'order_birdlife' if 'order_birdlife' in active_df.columns else 'order_birdtree'
    family_col = 'family_birdlife' if 'family_birdlife' in active_df.columns else 'family_birdtree'
    
    # --- DYNAMIC KEY FIX ---
    # We use the same dataset_id logic to prevent the KeyError crash
    ds_id = st.session_state.get('dataset_id', 'default')

    if order_col in active_df.columns and family_col in active_df.columns:
        st.write("Click on the rings to 'Drill Down' from Orders to Families.")
        
        exclude_keywords = ['dimorphism']

        # Filter Numeric Cols: Must be numbers AND not contain an excluded keyword
        numeric_params = [
            col for col in active_df.select_dtypes(include=np.number).columns.tolist() 
            if not any(key in col.lower() for key in exclude_keywords)
        ]


        col1, col2 = st.columns(2)
        with col1:
            # Added a unique key here!
            color_trait = st.selectbox(
                "Color Hierarchy by (Mean Value):", 
                options=[None] + numeric_params,
                key=f"sun_color_{ds_id}"
            )
        with col2:
            st.info("💡 Pro Tip: Sunburst charts are interactive. Click the inner rings to zoom in on a specific Order!")

        # 2. Build the Sunburst
        # Use a small sample or limit if the dataset is massive to prevent browser lag
        plot_df = active_df.dropna(subset=[order_col, family_col])
        
        if color_trait:
            plot_df = plot_df.dropna(subset=[color_trait])

        fig_sun = px.sunburst(
            plot_df,
            path=[order_col, family_col], 
            values=color_trait if color_trait else None,
            color=color_trait if color_trait else order_col,
            color_continuous_scale='Viridis',
            title=f"Taxonomic Distribution",
            template="plotly_white",
            # Added a key to the chart's internal state via the session ID
            ids=None 
        )
        
        fig_sun.update_layout(height=700, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)
        
    else:
        st.error("Taxonomic columns (Order/Family) not found in the current dataset.")