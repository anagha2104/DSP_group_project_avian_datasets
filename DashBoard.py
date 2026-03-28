import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# Make the dashboard utilize the full width of the monitor
st.set_page_config(page_title="Avian Explorer", layout="wide")


# 1. Load the AVONET dataset from the GitHub repository folder
@st.cache_data
def load_data():
    # Make sure this matches the exact file name your teammate uploaded
    df = pd.read_csv('data/raw/core/AVONET1_BirdLife.csv') 
    return df

df = load_data()

st.title("🦅 Avian Trait Database Explorer")

# --- CREATE TABS TO PREVENT RENDER LAG ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Trait Lookup", "🎯 Reverse Matcher", "🧬 PCA Analysis", "🌍 Global Map"])

# ==========================================
# TAB 1: BIRD TRAIT LOOKUP & RELATIONSHIPS
# ==========================================
with tab1:
    st.header("Bird Trait Lookup")

    bird_list = df['Species1'].unique()
    selected_bird = st.selectbox("Type or select a bird species to view its traits:", options=bird_list)


    bird_data = df[df['Species1'] == selected_bird]
    st.subheader(f"Data for: {selected_bird}")
    st.dataframe(bird_data)

    # Option B: Show a few specific traits beautifully (Optional but looks great!)
    st.write("### Key Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        # .iloc[0] grabs the first matching row's specific column value
        st.metric(label="Body Mass (g)", value=bird_data['Mass'].iloc[0]) 
    with col2:
        st.metric(label="Wing Length (mm)", value=bird_data['Wing.Length'].iloc[0])
    with col3:
        st.metric(label="Beak Length (mm)", value=bird_data['Beak.Length_Culmen'].iloc[0])

    st.write("---")
    st.write("### Trait Relationship Analysis")
    
    # 1. Setup dropdown menus for the user to choose traits
    # Replace these with the actual numeric column names from your group's CSV
    numeric_columns = ['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length'] 

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
        hover_data=['Species1'], # Lets the user see the bird's name when hovering!
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

# --- Assuming your dataframe is already loaded as 'df' ---
with tab2:

    st.header("🧬 Reverse Trait Matcher")
    st.write("Input specific measurements to find candidate species. Leave a trait unchecked to ignore it.")

    # 1. Define the traits we want to allow searching by
    # Ensure these match the exact column names in your team's CSV
    searchable_traits = ['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length']

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

        for trait, settings in active_filters.items():
            target = settings['target']
            tol_percent = settings['tolerance'] / 100.0
            min_bound = target - (target * tol_percent)
            max_bound = target + (target * tol_percent)
            results_df = results_df[(results_df[trait] >= min_bound) & (results_df[trait] <= max_bound)]

        st.subheader("🎯 Match Results")
        if len(active_filters) == 0:
            st.info("👈 Enable at least one filter above and click Run Search.")
        elif results_df.empty:
            st.warning("No birds found matching those exact parameters. Try increasing your +/- range!")
        else:
            st.success(f"Found {len(results_df)} matching species!")
            columns_to_show = ['Species1'] + list(active_filters.keys())
            st.dataframe(results_df[columns_to_show])
    
    
with tab3:
    st.header("🧬 PCA Cluster Analysis & Predictive Equations")
    st.write("Evaluate if linear combinations of physical traits can predict ecological categories.")

    all_numeric_traits = ['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length']
    categorical_factors = ['Migration', 'Habitat', 'Diet', 'Trophic.Level', 'Trophic.Niche','Primary.Lifestyle']

    col1, col2 = st.columns(2)
    with col1:
        selected_traits = st.multiselect("Select Traits for PCA:", options=all_numeric_traits, default=['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length'])
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
            hover_data=['Species1'] + selected_traits, 
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
    st.header("🌍 Global Avian Distribution")
    st.write("Visualizing species centroids and range sizes across biogeographical realms.")

    # 1. Clean the data for mapping
    # Maps will crash if they try to plot blank coordinates
    # Note: Double check the exact spelling of these columns in your CSV!
    map_df = df.dropna(subset=['Centroid.Latitude', 'Centroid.Longitude']).copy()

    # 2. Optional: Let the user color the map by a specific trait
    color_options = ['Migration', 'Habitat', 'Diet']
    available_colors = [c for c in color_options if c in df.columns]
    
    if available_colors:
        map_color = st.selectbox("Color map points by:", options=available_colors)
        map_df[map_color] = map_df[map_color].astype(str)
    else:
        map_color = None

    # 3. Create the Plotly Map
    fig_map = px.scatter_geo(
        map_df,
        lat="Centroid.Latitude",     # The column for up/down positioning
        lon="Centroid.Longitude",    # The column for left/right positioning
        hover_name="Species1",       # Shows the bird name when the user hovers
        size="Range.Size",           # Makes the dot larger if the bird has a huge range!
        color=map_color,             # Colors the dots based on the user's dropdown choice
        projection="natural earth",  # Gives the map a nice, slightly curved aesthetic
        template="plotly_white",
        title="Species Centroid Locations"
    )

    # 4. Styling tweaks
    # We set sizing limits so massive range sizes don't cover the entire map
    fig_map.update_traces(marker=dict(sizemin=2, sizemode='area'))
    fig_map.update_geos(showcountries=True, countrycolor="LightGrey")

    # 5. Display in Streamlit
    st.plotly_chart(fig_map, use_container_width=True)