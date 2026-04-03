import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import math
import plotly.graph_objects as go


# Make the dashboard utilize the full width of the monitor
st.set_page_config(page_title="Avian Explorer", layout="wide")


# 1. Load the AVONET dataset from the GitHub repository folder
@st.cache_data
def load_clean_data():
    # Make sure this matches the exact file name your teammate uploaded
    df = pd.read_csv('data/raw/avonet_FE_01.csv') 
    return df

@st.cache_data
def load_raw_data():
    # The original, messy dataset
    return pd.read_csv('data/raw/avonet_cleaned.csv')

@st.cache_data
def load_image_data():
    img_df = pd.read_csv('data/raw/bird_image_links.csv')
    return img_df

# Load both into memory immediately
df_clean = load_clean_data()
df_raw = load_raw_data()
image_df = load_image_data()

# --- GLOBAL STATE INITIALIZATION ---
if 'tab2_results' not in st.session_state:
    st.session_state['tab2_results'] = None
if 'working_df' not in st.session_state:
    st.session_state['working_df'] = df_clean

# --- SIDEBAR DATA MANAGER ---
st.sidebar.header("📁 Data Source Manager")
st.sidebar.write("Select the dataset to use for Analysis (Tabs 1, 3, & 4):")

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
    st.sidebar.info("💡 **Notice:** You are using uncleaned data. Watch how the PCA clusters shift or shrink in Tab 3 because the algorithm has to drop rows with missing values (NaNs)!")

elif data_source == "Tab 2 Filtered Results":
    if st.session_state['tab2_results'] is not None:
        st.session_state['working_df'] = st.session_state['tab2_results']
    else:
        st.sidebar.warning("⚠️ No filtered results yet. Run a search in Tab 2 first!")
        st.session_state['working_df'] = st.session_state['raw_df']

elif data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your own dataset (.csv)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            
            # --- STRICT SCHEMA VALIDATION ---
            # Define the exact columns your app needs to function
            required_columns = [
                'species_birdlife', 'mass', 'wing_len_avg', 'beak_culmen_avg', 'tarsus_avg', 'Centroid.Latitude', 'Centroid.Longitude'
            ]
            
            # Check if any are missing
            missing_cols = [col for col in required_columns if col not in custom_df.columns]
            
            if missing_cols:
                # Reject the upload and tell them exactly why
                st.sidebar.error(f"❌ Upload Rejected. Your CSV must follow the exact AVONET format. Missing columns: {', '.join(missing_cols)}")
            else:
                # Accept the upload
                st.session_state['working_df'] = custom_df
                st.sidebar.success("✅ Valid file uploaded and activated successfully!")
                
        except Exception as e:
            st.sidebar.error("Error reading the CSV file. Please ensure it is a valid text file.")

st.title("🦅 Avian Trait Database Explorer")

def calculate_overlap_percentage(lat1, lon1, r_user, lat2, lon2, area_bird):
    # 1. Haversine formula to find distance (d) between centers in km
    R_earth = 6371.0 
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R_earth * c
    
    # 2. Bird radius from area
    r_bird = np.sqrt(area_bird / np.pi)
    
    # 3. Intersection Logic
    if d >= (r_user + r_bird):
        return 0.0 # Too far apart
    elif d <= abs(r_user - r_bird):
        # One circle is entirely inside the other
        return (np.pi * min(r_user, r_bird)**2 / area_bird) * 100
    else:
        # Complex partial overlap formula
        part1 = r_user**2 * np.arccos((d**2 + r_user**2 - r_bird**2) / (2 * d * r_user))
        part2 = r_bird**2 * np.arccos((d**2 + r_bird**2 - r_user**2) / (2 * d * r_bird))
        part3 = 0.5 * np.sqrt((-d + r_user + r_bird) * (d - r_user + r_bird) * (d + r_user - r_bird) * (d + r_user + r_bird))
        intersection_area = part1 + part2 - part3
        
        return (intersection_area / area_bird) * 100

def generate_map_circle(lat, lon, radius_km, num_points=64):
    """Generates latitude and longitude points for a circle on the Earth's surface."""
    R = 6371.0 # Earth radius in km
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    d = radius_km / R

    circle_lats, circle_lons = [], []
    for i in range(num_points + 1):
        bearing = math.radians((i / num_points) * 360)
        new_lat = math.asin(math.sin(lat_rad) * math.cos(d) + 
                            math.cos(lat_rad) * math.sin(d) * math.cos(bearing))
        new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat_rad), 
                                       math.cos(d) - math.sin(lat_rad) * math.sin(new_lat))
        
        circle_lats.append(math.degrees(new_lat))
        circle_lons.append(math.degrees(new_lon))
        
    return circle_lats, circle_lons

# --- CREATE TABS TO PREVENT RENDER LAG ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Trait Lookup", "🎯 Reverse Matcher", "🧬 Trait Relationship Analysis and PCA", "🌍 Global Map"])

# ==========================================
# TAB 1: BIRD TRAIT LOOKUP & RELATIONSHIPS
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

    active_df = st.session_state['working_df']
    import pandas as pd # Ensure pandas is available to check for NaNs
    
   # --- 1. CREATE FAST LOOKUP DICTIONARIES (Bulletproof Method) ---
    
    # First, safely fill any missing names with Pandas
    safe_lib1 = active_df['species_birdlife'].fillna("Name Unavailable")
    safe_lib2 = active_df['species_birdtree'].fillna("Name Unavailable")
    
    # Now, explicitly lock them into dictionaries using dict(zip())
    id_to_lib1 = dict(zip(active_df['avibase_id'], safe_lib1))
    id_to_lib2 = dict(zip(active_df['avibase_id'], safe_lib2))
    
    bird_ids = active_df['avibase_id'].unique().tolist()

    # Initialize the central ID
    if 'current_bird_id' not in st.session_state:
        st.session_state['current_bird_id'] = bird_ids[0]

    # --- 2. THE CRITICAL FIX: FORCE-SYNCING CALLBACKS ---
    def update_from_lib1():
        # Update central ID
        st.session_state['current_bird_id'] = st.session_state['lib1_selector']
        # FORCE Dropdown 2 to update its memory to match Dropdown 1!
        st.session_state['lib2_selector'] = st.session_state['lib1_selector']
        
    def update_from_lib2():
        # Update central ID
        st.session_state['current_bird_id'] = st.session_state['lib2_selector']
        # FORCE Dropdown 1 to update its memory to match Dropdown 2!
        st.session_state['lib1_selector'] = st.session_state['lib2_selector']

    # --- 3. THE UI ---
    search_col1, search_col2 = st.columns(2)
    
    with search_col1:
        st.selectbox("Search by BirdLife Name:", 
                     options=bird_ids, 
                     format_func=lambda x: str(id_to_lib1.get(x, "Unknown")), 
                     key='lib1_selector', 
                     on_change=update_from_lib1)
                     
    with search_col2:
        st.selectbox("Search by BirdTree Name:", 
                     options=bird_ids, 
                     format_func=lambda x: str(id_to_lib2.get(x, "Unknown")), 
                     key='lib2_selector', 
                     on_change=update_from_lib2)

    st.write("---")

    # --- 4. THE DATA DISPLAY ---
    selected_id = st.session_state['current_bird_id']
    bird_data = active_df[active_df['species_birdlife'] == selected_id]
    
    profile_col1, profile_col2 = st.columns([1, 2])
    
     
    with profile_col1:
        # The image lookup is now 100% ID-based! 
        if 'scientific_name' in image_df.columns:
            matching_images = image_df[image_df['scientific_name'] == selected_id]
            if not matching_images.empty:
                img_url = matching_images['image_url'].iloc[0]
                st.image(img_url, caption=id_to_lib1[selected_id], use_container_width=True)
            else:
                st.info("📷 No image available.")
        else:
            st.error("Image database is missing 'species_birdlife'.")
            
    with profile_col2:
        # We grab the name safely to display in the Subheader
        display_name = id_to_lib1.get(selected_id)
        if display_name == "Name Unavailable": # Fallback to the other library if the first is blank
            display_name = id_to_lib2.get(selected_id, "Unknown Species")
            
        st.subheader(f"Data for: {display_name}")
        

    bird_list = active_df['species_birdlife'].unique()
    selected_bird = st.selectbox("Type or select a bird species to view its traits:", options=bird_list)


    bird_data = active_df[active_df['species_birdlife'] == selected_bird]

    st.write("---")
    
    # --- NEW: SPLIT LAYOUT FOR IMAGE AND METRICS ---
    # Create two columns: left for the image, right for the stats
    profile_col1, profile_col2 = st.columns([1, 2])
    
    with profile_col1:
        # Safely grab the display name from our dictionary using the ID
        display_name = id_to_lib1.get(selected_id, "Unknown Species")
        if display_name == "Name Unavailable":
            display_name = id_to_lib2.get(selected_id, "Unknown Species")
            
        # 1. The ID-based Image Logic (with fixed fallback)
        if 'avibase_id' in image_df.columns:
            matching_images = image_df[image_df['avibase_id'] == selected_id]
        else:
            # Fallback uses the display_name, since selected_bird no longer exists!
            matching_images = image_df[image_df['scientific_name'] == display_name]
            
        if not matching_images.empty:
            img_url = matching_images['image_url'].iloc[0]
            st.image(img_url, caption=display_name, use_container_width=True)
        else:
            st.info("📷 No image available.")
            
    with profile_col2:
        st.subheader(f"Data for: {display_name}")
        
        st.write("### Key Measurements")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        # We add dynamic checks here because your teammate made the columns lowercase in the new files!
        with m_col1:
            if 'mass' in active_df.columns: st.metric(label="Body Mass (g)", value=bird_data['mass'].iloc[0])
            elif 'mass' in active_df.columns: st.metric(label="Body Mass (g)", value=bird_data['mass'].iloc[0])
        with m_col2:
            if 'wing_len_avg' in active_df.columns: st.metric(label="Wing Length (mm)", value=bird_data['wing_len_avg'].iloc[0])
            elif 'wing_len_avg' in active_df.columns: st.metric(label="Wing Length (mm)", value=bird_data['wing_len_avg'].iloc[0])
        with m_col3:
            if 'beak_culmen_avg' in active_df.columns: st.metric(label="Beak Length (mm)", value=bird_data['beak_culmen_avg'].iloc[0])
            elif 'Beak.Length_Culmen' in active_df.columns: st.metric(label="Beak Length (mm)", value=bird_data['Beak.Length_Culmen'].iloc[0])
                
        st.write("") 
        st.subheader("Raw Data Row")
        st.dataframe(bird_data)

    # Option B: Show a few specific traits beautifully (Optional but looks great!)
    st.write("### Key Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        # .iloc[0] grabs the first matching row's specific column value
        st.metric(label="Body Mass (g)", value=bird_data['mass'].iloc[0]) 
    with col2:
        st.metric(label="Wing Length (mm)", value=bird_data['wing_len_avg'].iloc[0])
    with col3:
        st.metric(label="Beak Length (mm)", value=bird_data['Beak.Length_Culmen'].iloc[0])



# --- Assuming your dataframe is already loaded as 'df' ---
with tab2:
    st.header("🧬 Reverse Trait Matcher")
    
    # --- NEW SAFETY CHECK ---
    # This automatically finds the correct name column, no matter which dataset they select!
    display_name = 'species_birdlife' if 'species_birdlife' in active_df.columns else 'species_birdtree'

    # --- LIVE GEOSPATIAL PREVIEW (OUTSIDE THE FORM) ---
    st.subheader("🌍 1. Geographic Filter (Optional)")
    use_map_filter = st.checkbox("🔍 Enable Geographic Search")
    
    if use_map_filter:
        col_map1, col_map2 = st.columns([1, 2]) # Split screen: Controls on left, Map on right
        
        with col_map1:
            target_lat = st.slider("Target Latitude", min_value=-90.0, max_value=90.0, value=20.0, step=0.5)
            target_lon = st.slider("Target Longitude", min_value=-180.0, max_value=180.0, value=75.0, step=0.5)
            search_radius = st.slider("Search Radius (km)", min_value=100, max_value=5000, value=1000, step=100)
            min_overlap = st.slider("Min. Bird Range Overlap (%)", min_value=1, max_value=100, value=50)
            
        with col_map2:
            # 1. Generate the math points for our search circle
            circle_lats, circle_lons = generate_map_circle(target_lat, target_lon, search_radius)
            
            # 2. Draw a blank map focused on the user's coordinates
            fig_preview = go.Figure(go.Scattergeo())
            
            # 3. Add the search radius as a red circle
            fig_preview.add_trace(go.Scattergeo(
                lat=circle_lats,
                lon=circle_lons,
                mode='lines',
                line=dict(color='red', width=2),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)', # Semi-transparent red inside
                name='Search Area'
            ))
            
            # 4. Center the map dynamically
            fig_preview.update_geos(
                center=dict(lat=target_lat, lon=target_lon),
                projection_scale=3, # Zooms in nicely
                showcountries=True,
                showcoastlines=True,
                countrycolor="LightGrey"
            )
            fig_preview.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_preview, use_container_width=True)

    st.write("Input specific measurements to find candidate species. Leave a trait unchecked to ignore it.")

    # 1. Define the traits we want to allow searching by
    # Ensure these match the exact column names in your team's CSV
    searchable_traits = ['mass', 'wing_len_avg', 'beak_culmen_avg', 'tarsus_avg']

    # A dictionary to store only the active filters the user turns on
    active_filters = {}

    # WRAPPING INPUTS IN A FORM PREVENTS CONSTANT RE-RUNS
    with st.form("reverse_matcher_form"):
        for trait in searchable_traits:
            with st.container():
                if st.checkbox(f"🔍 Filter by {trait}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        target_val = st.number_input(f"Target {trait}", min_value=0.0, value=50.0, step=1.0, key=f"target_{trait}")
                    with c2:
                        tolerance = st.slider(f"+/- Range (%)", min_value=1, max_value=50, value=10, key=f"tol_{trait}")
                    
                    active_filters[trait] = {'target': target_val, 'tolerance': tolerance}
                st.write("") 
                
        # The math only triggers when this button is clicked
        submit_search = st.form_submit_button("Run Search")

    

    if submit_search:
        results_df = df.copy()

        # Physical trait filtering
        for trait, settings in active_filters.items():
            target = settings['target']
            tol_percent = settings['tolerance'] / 100.0
            min_bound = target - (target * tol_percent)
            max_bound = target + (target * tol_percent)
            results_df = results_df[(results_df[trait] >= min_bound) & (results_df[trait] <= max_bound)]

        # Add the Map Filter Logic (Now correctly indented!)
        if use_map_filter:
            # Drop rows missing map data so the math doesn't crash
            results_df = results_df.dropna(subset=['Centroid.Latitude', 'Centroid.Longitude', 'Range.Size'])
            
            # Apply our complex math function to every remaining bird
            overlaps = results_df.apply(
                lambda row: calculate_overlap_percentage(
                    target_lat, target_lon, search_radius,
                    row[lat_col], row['Centroid.Longitude'], row['Range.Size']
                ), axis=1
            )
            
            # Filter the dataframe keeping only birds that meet the overlap threshold
            results_df['Overlap_%'] = overlaps
            results_df = results_df[results_df['Overlap_%'] >= min_overlap]

        st.subheader("🎯 Match Results")
        
        # --- THE FIX IS ON THIS NEXT LINE ---
        if len(active_filters) == 0 and not use_map_filter:
            st.info("👈 Enable at least one filter above (Geographic or Trait) and click Run Search.")
            
        elif results_df.empty:
            st.warning("No birds found matching those exact parameters. Try increasing your +/- range or decreasing the required map overlap!")
            
        else:
            # --- NEW: Save to Session State ---
            st.session_state['tab2_results'] = results_df
            st.info("💡 **These results have been saved to your workspace!** Select 'Tab 2 Filtered Results' in the left sidebar to run PCA or Trait Analysis on this specific list of birds.")
            # ----------------------------------
            
            st.success(f"Found {len(results_df)} matching species!")
                       
            # Build the columns we want to show in the final table
            # Show both names to the user in the final table, then the traits
            columns_to_show = ['species_birdlife', 'species_birdtree'] + list(active_filters.keys())
            
            if use_map_filter:
                # Round the overlap percentage to 2 decimal places so it looks clean
                results_df['Overlap_%'] = results_df['Overlap_%'].round(2)
                columns_to_show.append('Overlap_%')
                
            st.dataframe(results_df[columns_to_show])
            
            # --- DOWNLOAD BUTTONS ---
            st.write("---")
            st.write("### 📥 Export Data")
            
            # Place the buttons side-by-side to make the UI look clean
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                # Button 1: Just the columns shown on the screen
                csv_summary = results_df[columns_to_show].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Summary (Visible Columns)",
                    data=csv_summary,
                    file_name='bird_match_summary.csv',
                    mime='text/csv',
                )
                
            with dl_col2:
                # Button 2: EVERY column from the original dataset for the matched birds
                csv_full = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📦 Download Full Dataset (All AVONET Columns)",
                    data=csv_full,
                    file_name='bird_match_full_data.csv',
                    mime='text/csv',
                )

            # --- 4. MAP OF RESULTS ---
            st.write("---")
            st.subheader("🗺️ Geographic Distribution of Matches")

            # --- THE NEW DYNAMIC WARNING ---
            if use_map_filter and min_overlap < 35:
                st.info("💡 **Note on Map Visuals:** Because you selected a low overlap requirement, some matched birds may have center dots that appear outside your red search circle. This happens when a bird has a massive continental range that reaches into your search area, even if its exact center is far away!")

            
            # Create a fresh Plotly figure for the results
            fig_results = go.Figure()

            # Layer 1: Draw the user's search region (Only if they enabled the map filter)
            if use_map_filter:
                fig_results.add_trace(go.Scattergeo(
                    lat=circle_lats,
                    lon=circle_lons,
                    mode='lines',
                    line=dict(color='red', width=2),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)', # Faint red background
                    name='Your Search Region'
                ))

            # Layer 2: Draw the matched birds
            # First, drop any birds that might be missing coordinate data so the map doesn't crash
            map_results_df = results_df.dropna(subset=['Centroid.Latitude', 'Centroid.Longitude', 'Range.Size'])

            if not map_results_df.empty:
                # Math trick: Scale the dot sizes so huge geographic ranges don't cover the 
                # whole screen, but tiny island birds don't become invisible.
                max_range = map_results_df['Range.Size'].max()
                marker_sizes = (map_results_df['Range.Size'] / max_range * 40) + 6 if max_range > 0 else 8

                fig_results.add_trace(go.Scattergeo(
                    lat=map_results_df['Centroid.Latitude'],
                    lon=map_results_df['Centroid.Longitude'],
                    # --- NEW: Added Overlap % and Range Size to the hover text ---
                    customdata=np.stack((map_results_df['Overlap_%'], map_results_df['Range.Size']), axis=-1),
                    hovertemplate="<b>%{customdata[2]}</b><br><i>%{text}</i><br>Distance from Center: %{customdata[0]:.1f} km<br>Range: %{customdata[1]:,.0f} sq km<extra></extra>",
                    text=map_results_df['species_birdlife'],
                    mode='markers',
                    marker=dict(
                        size=marker_sizes,
                        color='#3b82f6',
                        opacity=0.7,
                        line=dict(width=1, color='black')
                    ),
                    name='Matched Birds'
                ))

            # Format the globe
            fig_results.update_geos(
                showcountries=True,
                showcoastlines=True,
                countrycolor="LightGrey",
                projection_type="natural earth"
            )

            # UX Upgrade: If the user searched a specific area, automatically zoom the camera in on that region!
            if use_map_filter:
                fig_results.update_geos(
                    center=dict(lat=target_lat, lon=target_lon),
                    projection_scale=2.5 
                )

            st.plotly_chart(fig_results, use_container_width=True)       
    
    
    
with tab3:
        
    st.write("### Trait Relationship Analysis")
    
    # 1. Setup dropdown menus for the user to choose traits
    # Replace these with the actual numeric column names from your group's CSV
    numeric_columns = ['mass', 'wing_len_avg', 'beak_culmen_avg', 'tarsus_avg'] 

    col1, col2 = st.columns(2)
    with col1:
        x_trait = st.selectbox("Select X-Axis Trait:", options=numeric_columns, index=0)
    with col2:
        y_trait = st.selectbox("Select Y-Axis Trait:", options=numeric_columns, index=1)

    # 2. Clean the data (Math functions will crash if there are blank/NaN values)
    clean_df = df.dropna(subset=[x_trait, y_trait])

    # 3. Extract the x and y columns as numpy arrays
    x_vals = clean_df[x_trait]
    y_vals = clean_df[y_trait]

    # 4. Calculate the Line of Best Fit (Linear Regression)
    # Degree 1 means a straight line. It returns the slope (m) and intercept (c)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    # 5. Calculate the Correlation Coefficient (r) to see how strong the relationship is
    correlation_matrix = np.corrcoef(x_vals, y_vals)
    r_value = correlation_matrix[0, 1]

    # 6. Display the Equation and Correlation to the user
    st.success("Mathematical Relationship")

    # Using LaTeX formatting to make the equation look like a real math formula
    st.latex(rf"{y_trait} = {slope:.3f} \cdot {x_trait} + {intercept:.3f}")

    st.write(f"**Pearson Correlation (r):** {r_value:.3f}")

    # --- NEW PLOTLY CHART FOR TAB 1 ---
    fig1 = px.scatter(
        clean_df, 
        x=x_trait, 
        y=y_trait,
        hover_data=['species_birdlife', 'species_birdtree'],
        title=f"Relationship: {y_trait} vs {x_trait}",
        template="plotly_white"  # Gives it a clean, modern look
    )
    
    # Add the line of best fit visually to the chart
    fig1.add_scatter(
        x=x_vals, 
        y=(slope * x_vals + intercept), 
        mode='lines', 
        name='Trendline',
        line=dict(color='red', dash='dash')
    )
    
    st.plotly_chart(fig1, use_container_width=True)


    st.header("🧬 PCA Cluster Analysis & Predictive Equations")
    st.write("Evaluate if linear combinations of physical traits can predict ecological categories.")

    all_numeric_traits = ['mass', 'wing_len_avg', 'beak_culmen_avg', 'tarsus_avg']
    categorical_factors = ['Migration', 'Habitat', 'Diet', 'Trophic.Level', 'Trophic.Niche','Primary.Lifestyle']

    col1, col2 = st.columns(2)
    with col1:
        selected_traits = st.multiselect("Select Traits for PCA:", options=all_numeric_traits, default=['mass', 'wing_len_avg', 'beak_culmen_avg', 'tarsus_avg'])
    with col2:
        available_factors = [f for f in categorical_factors if f in df.columns] 
        if available_factors:
            grouping_factor = st.selectbox("Select Category to Color By:", options=available_factors)
        else:
            st.warning("Grouping columns not found in dataset. Check exact column names.")
            grouping_factor = None

    if len(selected_traits) >= 2 and grouping_factor:
        columns_to_keep = selected_traits + [grouping_factor]
        pca_df = df.dropna(subset=columns_to_keep).copy()
        pca_df[grouping_factor] = pca_df[grouping_factor].astype(str)

        X = pca_df[selected_traits].values
        
        # 1. Standardize Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ==========================================
        # EXPLICIT CORRELATION MATRIX METHOD
        # ==========================================
        
        # 2. Calculate the Correlation Matrix
        # We transpose X_scaled because np.corrcoef expects variables as rows
        corr_matrix = np.corrcoef(X_scaled.T)

        # 3. Perform Eigendecomposition
        # This extracts the eigenvalues (variance explained) and eigenvectors (the axes)
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

        # 4. Sort Components
        # We must sort them so PC1 is the axis with the highest eigenvalue
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Project the Data
        # We take our scaled data and multiply (dot product) it by the top 2 eigenvectors
        projection_matrix = sorted_eigenvectors[:, 0:2]
        X_pca = X_scaled.dot(projection_matrix)

        pca_df['PC1'] = X_pca[:, 0]
        pca_df['PC2'] = X_pca[:, 1]

        # 6. Extract Loadings for the Equations
        # Transpose so it matches the shape expected by our equation builder
        loadings = projection_matrix.T
        
        # ==========================================

        def build_equation(pc_index):
            terms = []
            for i, trait in enumerate(selected_traits):
                weight = loadings[pc_index, i]
                sign = "+" if weight >= 0 else ""
                terms.append(f"{sign}{weight:.3f}({trait})")
            return " ".join(terms)

        eq_pc1 = build_equation(0)
        eq_pc2 = build_equation(1)

        st.subheader("Linear Combinations (The Equations)")
        st.write("*(Note: Equations use scaled data).*")
        st.latex(rf"PC_1 = {eq_pc1}")
        st.latex(rf"PC_2 = {eq_pc2}")

        # Calculate Explained Variance explicitly from eigenvalues
        total_variance = np.sum(sorted_eigenvalues)
        var_pc1 = (sorted_eigenvalues[0] / total_variance) * 100
        var_pc2 = (sorted_eigenvalues[1] / total_variance) * 100
        
        st.caption(f"**Variance Explained:** PC1 ({var_pc1:.1f}%) | PC2 ({var_pc2:.1f}%) | Total ({(var_pc1 + var_pc2):.1f}%)")

        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=grouping_factor,
            hover_data=['species_birdlife'] + selected_traits, 
            title=f"PCA Clustering by {grouping_factor}",
            template="plotly_white"
        )
        
        fig_pca.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.info(f"**How to read this:** Look at the colors ({grouping_factor}). If the colors naturally form distinct, separate clumps on the graph, then YES, the linear equations above can accurately predict a bird's {grouping_factor}.")
    else:
        st.warning("⚠️ Please select at least two traits and a valid grouping factor.")

# ==========================================
# TAB 4: BIOGEOGRAPHICAL MAP
# ==========================================
with tab4:
    st.header("Global Distribution Map")
    active_df = st.session_state['working_df']
    
    # --- Dynamic Column Check for new lowercase names ---
    lat_col = 'lat_centroid' if 'lat_centroid' in active_df.columns else 'Centroid.Latitude'
    lon_col = 'lon_centroid' if 'lon_centroid' in active_df.columns else 'Centroid.Longitude'
    range_col = 'range_size' if 'range_size' in active_df.columns else 'Range.Size'

    st.write("Visualize the geographic center of the selected datasets.")

    # 1. Clean the data (Using our dynamic columns)
    map_df = active_df.dropna(subset=[lat_col, lon_col]).copy()

    # 2. Optional: Let the user color the map by a specific trait
    color_options = ['migration', 'habitat', 'primary_diet', 'Migration', 'Habitat', 'Diet']
    available_colors = [c for c in color_options if c in active_df.columns]
    
    if available_colors:
        map_color = st.selectbox("Color map points by:", options=available_colors)
        map_df[map_color] = map_df[map_color].astype(str)
    else:
        map_color = None

    # 3. Create the Plotly Map
    fig_map = px.scatter_geo(
        map_df,
        lat=lat_col,                 # Now dynamically picks up lat_centroid
        lon=lon_col,                 # Now dynamically picks up lon_centroid
        hover_name="species_birdlife",       
        size=range_col,              # Now dynamically picks up range_size
        color=map_color,             
        projection="natural earth",  
        template="plotly_white",
        title="Species Centroid Locations",
        
        hover_data={
            "species_birdtree": True, 
            range_col: True, 
            lat_col: False, 
            lon_col: False
        }
    )

    # 4. Styling tweaks
    fig_map.update_traces(marker=dict(sizemin=2, sizeref=20))
    st.plotly_chart(fig_map, use_container_width=True)