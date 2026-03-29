import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from django.shortcuts import render
from .models import BirdTrait
import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance in km between two lat/lon points on Earth."""
    R_earth = 6371.0 # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_earth * c

def circle_intersection_area(d, r1, r2):
    """Calculates the overlapping area of two circles."""
    if d >= r1 + r2:
        return 0.0 # No overlap
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2 # One circle is completely inside the other
    
    # Partial overlap math
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    
    area1 = r1_sq * math.acos(d1 / r1) - d1 * math.sqrt(r1_sq - d1**2)
    area2 = r2_sq * math.acos(d2 / r2) - d2 * math.sqrt(r2_sq - d2**2)
    return area1 + area2

def dashboard_home(request):
    total_birds = BirdTrait.objects.count()
    
    # 1. Define the traits we want to allow the user to select
    # The dictionary maps database field names to human-readable labels
    available_traits = {
        'mass': 'Body Mass (g)',
        'wing_length': 'Wing Length (mm)',
        'beak_length': 'Beak Length (mm)',
        'tarsus_length': 'Tarsus Length (mm)'
    }

    # 2. Capture user input from the URL (the GET request)
    # request.GET.get('key', 'default_value')
    x_trait = request.GET.get('x_trait', 'mass')
    y_trait = request.GET.get('y_trait', 'wing_length')

    # Security check: ensure they didn't tamper with the URL to request a non-existent column
    if x_trait not in available_traits: x_trait = 'mass'
    if y_trait not in available_traits: y_trait = 'wing_length'

    # 3. Fetch data dynamically based on their selection
    # **exclude_kwargs unpacks the dictionary into the Django query to drop missing values
    exclude_kwargs = {
        f"{x_trait}__isnull": True,
        f"{y_trait}__isnull": True
    }
    
    bird_data = BirdTrait.objects.exclude(**exclude_kwargs).values(
        'species_name', x_trait, y_trait,
    )[:1000]

    df = pd.DataFrame(list(bird_data))

    # 4. Create the Plotly Figure
    chart_html = ""
    math_stats = None

    if not df.empty:
        x_vals = df[x_trait]
        y_vals = df[y_trait]

        # 1. Calculate the math (Line of Best Fit & Correlation)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        r_matrix = np.corrcoef(x_vals, y_vals)
        r_value = r_matrix[0, 1]
        r_squared = r_value ** 2  # R-squared shows how well the line fits the data

        # Pack the stats into a dictionary, rounding for a clean display
        math_stats = {
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'r_value': round(r_value, 4),
            'r_squared': round(r_squared, 4)
        }

        # 2. Create the base scatter plot (Removed color parameter)
        fig = px.scatter(
            df, 
            x=x_trait, 
            y=y_trait,
            hover_data=['species_name'],
            title=f"Relationship: {available_traits[y_trait]} vs {available_traits[x_trait]}",
            template="plotly_white",
            labels={
                x_trait: available_traits[x_trait],
                y_trait: available_traits[y_trait]
            }
        )

        # 3. Add the trendline visually to the chart
        fig.add_scatter(
            x=x_vals, 
            y=(slope * x_vals + intercept), 
            mode='lines', 
            name='Trendline (OLS)',
            line=dict(color='red', dash='dash')
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    
    # --- Single Bird Lookup Logic ---
    searched_bird = None
    search_query = request.GET.get('species_search', '')
    if search_query:
        searched_bird = BirdTrait.objects.filter(species_name__icontains=search_query).first()

    # NEW: Grab a fast, flat list of every single bird name in the database
    all_species = BirdTrait.objects.values_list('species_name', flat=True)

    # 5. Pack EVERYTHING into the context dictionary
    context = {
        'total_birds': total_birds,
        'chart': chart_html,
        'available_traits': available_traits,
        'selected_x': x_trait,
        'selected_y': y_trait,
        'math_stats': math_stats,
        'searched_bird': searched_bird,  # Send the found bird to HTML
        'search_query': search_query,    # Send their text back so the search bar doesn't clear
        'all_species': all_species,
    }
    
    return render(request, 'dashboard/home.html', context)
def reverse_matcher(request):
    results = None
    traits = {
        'mass': 'Body Mass',
        'wing_length': 'Wing Length',
        'beak_length': 'Beak Length',
        'tarsus_length': 'Tarsus Length'
    }
    
    if request.GET.get('search'):
        query = BirdTrait.objects.all()
        
        # 1. Apply the standard physical trait filters first
        for key in traits.keys():
            if request.GET.get(f'check_{key}'):
                target = float(request.GET.get(f'target_{key}', 0))
                tol = float(request.GET.get(f'tol_{key}', 10)) / 100.0
                query = query.filter(**{f'{key}__gte': target * (1 - tol), f'{key}__lte': target * (1 + tol)})
        
        # 2. Convert the Django QuerySet to a list to perform Python math
        filtered_birds = list(query)
        
        # 3. Apply the Spatial Filter if checked
        if request.GET.get('check_spatial'):
            user_lat = float(request.GET.get('user_lat', 0))
            user_lon = float(request.GET.get('user_lon', 0))
            user_radius = float(request.GET.get('user_radius', 1000)) # in km
            min_overlap_pct = float(request.GET.get('overlap_pct', 50)) / 100.0
            
            spatial_results = []
            for bird in filtered_birds:
                # Skip if bird has no location data
                if bird.centroid_latitude is None or bird.range_size is None:
                    continue
                
                # Bird radius = sqrt(Area / pi)
                bird_radius = math.sqrt(bird.range_size / math.pi)
                
                d = haversine(user_lat, user_lon, bird.centroid_latitude, bird.centroid_longitude)
                overlap_area = circle_intersection_area(d, user_radius, bird_radius)
                
                # Check if overlap meets the user's required threshold
                if overlap_area >= (min_overlap_pct * bird.range_size):
                    spatial_results.append(bird)
            
            results = spatial_results[:200]
        else:
            results = filtered_birds[:200]

    return render(request, 'dashboard/matcher.html', {'results': results, 'traits': traits})
def pca_analysis(request):
    numeric_traits = {
        'mass': 'Body Mass',
        'wing_length': 'Wing Length',
        'beak_length': 'Beak Length',
        'tarsus_length': 'Tarsus Length'
    }
    
    categorical_factors = {
        'habitat': 'Habitat',
        'diet': 'Diet',
        'migration': 'Migration',
        'trophic_level': 'Trophic Level',
        'primary_lifestyle': 'Primary Lifestyle'
    }

    selected_traits = request.GET.getlist('traits')
    if not selected_traits:
        selected_traits = list(numeric_traits.keys())
        
    grouping_factor = request.GET.get('factor', 'habitat')
    
    # NEW: Get the PCA method from the user, default to correlation
    pca_method = request.GET.get('pca_method', 'correlation')

    exclude_kwargs = {f"{trait}__isnull": True for trait in selected_traits}
    exclude_kwargs[f"{grouping_factor}__isnull"] = True

    query_fields = ['species_name', grouping_factor] + selected_traits
    bird_data = BirdTrait.objects.exclude(**exclude_kwargs).values(*query_fields)[:2000]

    df = pd.DataFrame(list(bird_data))

    chart_html = ""
    context = {
        'numeric_traits': numeric_traits,
        'categorical_factors': categorical_factors,
        'selected_traits': selected_traits,
        'selected_factor': grouping_factor,
        'pca_method': pca_method, # NEW: Send the chosen method back to the template
    }

    if not df.empty and len(selected_traits) >= 2:
        X = df[selected_traits]
        
        # NEW: Adjust the scaler based on the user's choice
        if pca_method == 'correlation':
            # Standardize (Mean=0, Variance=1) -> Mathematically equivalent to Correlation PCA
            scaler = StandardScaler(with_mean=True, with_std=True)
            method_title = "Correlation Matrix (Standardized)"
        else:
            # Mean-Center Only (Mean=0, Variance=Original) -> Mathematically equivalent to Covariance PCA
            scaler = StandardScaler(with_mean=True, with_std=False)
            method_title = "Covariance Matrix (Mean-Centered)"

        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
       
        df['PC1'] = components[:, 0]
        df['PC2'] = components[:, 1]

        loadings = pca.components_

        # Helper function to build the equation string
        def build_equation(pc_index):
            terms = []
            for i, trait in enumerate(selected_traits):
                weight = loadings[pc_index, i]
                sign = " + " if weight >= 0 and i > 0 else " " if weight >= 0 else " - "
                terms.append(f"{sign}{abs(weight):.3f}({numeric_traits[trait]})")
            return "".join(terms)

        context['eq_pc1'] = build_equation(0)
        context['eq_pc2'] = build_equation(1)
        
        var_pc1 = pca.explained_variance_ratio_[0] * 100
        var_pc2 = pca.explained_variance_ratio_[1] * 100
        context['var_pc1'] = round(var_pc1, 1)
        context['var_pc2'] = round(var_pc2, 1)
        context['total_var'] = round(var_pc1 + var_pc2, 1)

        # Build the Plotly Chart
        fig = px.scatter(
            df, x='PC1', y='PC2', color=grouping_factor,
            hover_data=['species_name'] + selected_traits,
            title=f"PCA Clustering by {categorical_factors.get(grouping_factor, grouping_factor)}",
            template="plotly_white"
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        
        context['chart'] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render(request, 'dashboard/pca.html', context)