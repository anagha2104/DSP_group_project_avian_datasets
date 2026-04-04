# Flask Advanced Analysis Dashboard

A comprehensive Flask application for exploratory data analysis of the AviNet bird dataset with interactive visualizations.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   python app.py
   ```

3. Open your browser and go to:
   - `http://127.0.0.1:5000/` - Home page
   - `http://127.0.0.1:5000/section1` - Advanced Analysis Dashboard

## Features

### Section 1 - Advanced Analysis Dashboard

A three-panel interactive analysis tool:

**Left Panel: Filters**
- Filter species by Order, Family, Trophic Niche, Habitat Density
- View active filters as removable badges
- Clear all filters with one click
- Real-time species count (filtered vs. total)

**Middle Panel: Analysis Controls & Statistics**
- Select from numerical or categorical features
- Log scale toggle (for numerical features only)
- Optional sub-feature selection for grouped analysis
- Dynamic statistics display:
  - **Numerical**: Mean, Median, Min, Max, Q1, Q3
  - **Categorical**: Unique count, Mode, Missing %, Top 10 categories

**Right Panel: Visualizations**
- **Numerical Features:**
  - Plot 1: Histogram + KDE curve (with optional log scale)
  - Plot 2: Binned bar chart (appears when sub-feature selected)
- **Categorical Features:**
  - Plot 1: Horizontal bar chart (sorted by count with percentages)
  - Plot 2: Stacked bar chart (appears when sub-feature selected)

### Data

Uses the AviNet dataset (`data/avonet.csv`):
- **11,237** bird species with **56** morphological and ecological traits
- Traits are classified as:
  - **Numerical**: tarsus, wing_len, tail, mass, log_mass, hwi, beak_depth
  - **Categorical**: habitat_density, habitat, lifestyle, trophic_level

### Sub-Feature Options

- Order (Taxonomic classification)
- Trophic Niche (Feeding guild)
- Habitat Density (Density of habitat)
- Migration (Migration status)

## Files

- `app.py` - Flask application with API routes
- `analysis_01.py` - Analysis functions and data processing
- `data/avonet.csv` - Bird species dataset
- `templates/section1.html` - Dashboard HTML
- `static/js/section1.js` - Dashboard interactivity
- `static/js/plotly.min.js` - Local Plotly.js library (no CDN needed)
- `requirements.txt` - Python dependencies

## Technical Details

- **Backend**: Flask REST API (`/api/analysis`, `/api/filter-values/`)
- **Frontend**: Vanilla JavaScript with Fetch API
- **Plotting**: Plotly.js (local copy)
- **Data processing**: Pandas, NumPy, SciPy
- **Performance**: Dataset loaded at app startup for instant analysis

## Troubleshooting

- If you get errors about missing modules, run: `pip install -r requirements.txt`
- If port 5000 is in use, Flask will use a different port (check terminal output)
- Ensure `data/avonet.csv` exists in the project directory