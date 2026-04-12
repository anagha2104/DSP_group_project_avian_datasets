# 🦅 Avian Trait Database Explorer

An interactive Data Science dashboard built with **Streamlit** to explore, filter, and analyze avian morphological traits and biogeographical data. This tool is specifically optimized for the **AVONET** dataset but supports custom CSV uploads for comparative analysis.

## ✨ Features

* **🔍 Trait Lookup:** Search for specific species by BirdLife or BirdTree nomenclature to view key measurements (Mass, Wing, Beak) and species imagery.
* **🎯 Reverse Matcher:** Find species based on physical dimensions with adjustable tolerance ranges and geographic radius filtering.
* **📊 Exploratory Data Analysis:** Visualize trait distributions via histograms, boxplots categorized by habitat or diet, and interactive **Principal Component Analysis (PCA)**.
* **🌍 3D Biogeographical Globe:** A 3D orthographic visualization of species distribution with markers scaled by range size.
* **🌳 Taxonomic Explorer:** Interactive sunburst charts to drill down through avian taxonomy from Order to Family.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/avian-explorer.git](https://github.com/your-username/avian-explorer.git)
   cd avian-explorer

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the Streamlit application:
streamlit run DashBoard_final_presentation.py

🛠️ Built With

    Streamlit - Web framework and UI state management

    Pandas & NumPy - Data manipulation, cleaning, and geospatial calculations

    Scikit-Learn - Dimensionality reduction (PCA) and data scaling (StandardScaler)

    Plotly - Interactive data visualization (3D globes, sunburst charts, and scatter plots)