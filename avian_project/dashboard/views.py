import pandas as pd
import numpy as np
import plotly.express as px
from django.shortcuts import render
from .models import BirdTrait

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

    # 5. Pack EVERYTHING into the context dictionary, including their current selection
    context = {
        'total_birds': total_birds,
        'chart': chart_html,
        'available_traits': available_traits,
        'selected_x': x_trait,
        'selected_y': y_trait,
        'math_stats': math_stats,
    }
    
    return render(request, 'dashboard/home.html', context)