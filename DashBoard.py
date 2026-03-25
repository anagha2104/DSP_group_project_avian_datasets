import streamlit as st
import pandas as pd
import numpy as np

# 1. Load the AVONET dataset from the GitHub repository folder
@st.cache_data
def load_data():
    # Make sure this matches the exact file name your teammate uploaded
    df = pd.read_csv('data/raw/core/AVONET1_BirdLife.csv') 
    return df

df = load_data()

st.title("🔍 Bird Trait Lookup")

# 2. Create the searchable dropdown menu
# Note: Check if your teammate's CSV uses 'Species', 'Species1', or something else.
bird_list = df['Species1'].unique()

selected_bird = st.selectbox(
    "Type or select a bird species to view its traits:",
    options=bird_list
)

# 3. Filter the dataset based on the user's selection
bird_data = df[df['Species1'] == selected_bird]

# 4. Display the results
st.subheader(f"Data for: {selected_bird}")

# Option A: Show the raw table row
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

import streamlit as st
import pandas as pd

# --- Assuming your dataframe is already loaded as 'df' ---

st.write("---")
st.header("🧬 Reverse Trait Matcher")
st.write("Input specific measurements to find candidate species. Leave a trait unchecked to ignore it.")

# 1. Define the traits we want to allow searching by
# Ensure these match the exact column names in your team's CSV
searchable_traits = ['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length']

# A dictionary to store only the active filters the user turns on
active_filters = {}

# 2. Build the dynamic UI
for trait in searchable_traits:
    # A visual container to keep each trait's controls grouped together
    with st.container():
        # The Checkbox: If checked, show the inputs. If unchecked, ignore this trait.
        if st.checkbox(f"🔍 Filter by {trait}"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Target Value Input
                target_val = st.number_input(
                    f"Target {trait}", 
                    min_value=0.0, 
                    value=50.0, 
                    step=1.0, 
                    key=f"target_{trait}" # Unique key required by Streamlit
                )
            
            with col2:
                # Tolerance Slider (%)
                tolerance = st.slider(
                    f"+/- Range (%)", 
                    min_value=1, 
                    max_value=50, 
                    value=10, 
                    key=f"tol_{trait}"
                )
            
            # Save the user's choices into our dictionary
            active_filters[trait] = {'target': target_val, 'tolerance': tolerance}
        
        st.write("") # Adds a tiny bit of vertical spacing between traits

# 3. Apply the filters to the Dataframe
results_df = df.copy()

# Loop through only the traits the user actually checked
for trait, settings in active_filters.items():
    target = settings['target']
    tol_percent = settings['tolerance'] / 100.0
    
    # Calculate the exact min and max boundaries based on the % slider
    min_bound = target - (target * tol_percent)
    max_bound = target + (target * tol_percent)
    
    # Overwrite the dataframe with only the rows that fall inside these boundaries
    results_df = results_df[(results_df[trait] >= min_bound) & (results_df[trait] <= max_bound)]

# 4. Display the Results
st.subheader("🎯 Match Results")

if len(active_filters) == 0:
    st.info("👈 Enable at least one filter above to start searching.")
elif results_df.empty:
    st.warning("No birds found matching those exact parameters. Try increasing your +/- range!")
else:
    st.success(f"Found {len(results_df)} matching species!")
    
    # We only display the 'Species' column PLUS the traits the user actually filtered by
    columns_to_show = ['Species1'] + list(active_filters.keys())
    st.dataframe(results_df[columns_to_show])

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Assuming your dataframe is already loaded as 'df' ---

st.write("---")
st.header("🧬 PCA Cluster Analysis & Predictive Equations")
st.write("Evaluate if linear combinations of physical traits can predict ecological categories.")

# 1. Inputs
all_numeric_traits = ['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length']
categorical_factors = ['Migration', 'Habitat', 'Diet', 'Trophic.Level']

col1, col2 = st.columns(2)
with col1:
    selected_traits = st.multiselect("Select Traits for PCA:", options=all_numeric_traits, default=['Mass', 'Wing.Length', 'Beak.Length_Culmen', 'Tarsus.Length'])
with col2:
    grouping_factor = st.selectbox("Select Category to Color By:", options=categorical_factors)

if len(selected_traits) >= 2:
    # 2. Clean and Scale Data
    columns_to_keep = selected_traits + [grouping_factor]
    pca_df = df.dropna(subset=columns_to_keep).copy()
    pca_df[grouping_factor] = pca_df[grouping_factor].astype(str)

    X = pca_df[selected_traits]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Run PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df['PC1'] = components[:, 0]
    pca_df['PC2'] = components[:, 1]

    # 4. Extract Loadings to build the Equations
    # pca.components_ contains the weights for how much each trait contributes to each PC
    loadings = pca.components_
    
    def build_equation(pc_index):
        terms = []
        for i, trait in enumerate(selected_traits):
            weight = loadings[pc_index, i]
            # Format to 3 decimal places. Add a '+' if positive.
            sign = "+" if weight >= 0 else ""
            terms.append(f"{sign}{weight:.3f}({trait})")
        return " ".join(terms)

    eq_pc1 = build_equation(0)
    eq_pc2 = build_equation(1)

    # 5. Display the Mathematical Equations
    st.subheader("Linear Combinations (The Equations)")
    st.write("These equations represent how your selected traits are mathematically combined to create the X and Y axes of the plot below. *(Note: Equations use scaled data).*")
    
    st.latex(rf"PC_1 = {eq_pc1}")
    st.latex(rf"PC_2 = {eq_pc2}")

    # 6. Display Explained Variance
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    st.caption(f"**Variance Explained:** PC1 ({var_pc1:.1f}%) | PC2 ({var_pc2:.1f}%) | Total ({(var_pc1 + var_pc2):.1f}%)")

    # 7. Draw the Interactive Chart to Check Groupings
    st.scatter_chart(
        data=pca_df,
        x='PC1',
        y='PC2',
        color=grouping_factor
    )
    
    # 8. Analytical Conclusion Prompt
    st.info(f"**How to read this:** Look at the colors ({grouping_factor}). If the colors naturally form distinct, separate clumps on the graph, then YES, the linear equations above can accurately predict a bird's {grouping_factor}. If the colors are completely mixed together like confetti, the physical traits do not strongly predict it.")

else:
    st.warning("⚠️ Please select at least two traits.")