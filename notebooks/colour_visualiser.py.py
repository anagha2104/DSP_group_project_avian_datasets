import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load bird color proportions CSV
birds_df = pd.read_csv("/home/anagha/DSP_group_project_avian_datasets/data/raw/colour/Data_S1/Information_for_Illustrations_and_proportion_of_24_colors.csv")

# Load RGB color mapping CSV
rgb_df = pd.read_csv("~/DSP_group_project_avian_datasets/data/raw/colour/Data_S1/RGB_values_for_color_classification.csv")  

# Function to get top N colors for a species
def get_top_colors(species_name, df, top_n=10):
    row = df[df['Com_name'] == species_name].iloc[0]
    color_cols = [f'color{i}' for i in range(1, 25)]
    colors = row[color_cols]
    top_colors = colors.sort_values(ascending=False).head(top_n)
    return top_colors

# Function to show color patches (modified slightly to accept a list of color names)
def show_color_patches_rgb(color_names, rgb_df):
    n = len(color_names)
    fig, axes = plt.subplots(n, 1, figsize=(5, n*1.2))
    
    if n == 1:
        axes = [axes]
    
    for ax, color_name in zip(axes, color_names):
        # Get RGB for this color
        rgb_row = rgb_df[rgb_df['Colors'] == color_name].iloc[0]

        try:
            r = float(rgb_row['R'])
            g = float(rgb_row['G'])
            b = float(rgb_row['B'])
        except ValueError:
            print(f"Invalid RGB for color {color_name}, skipping...")
            continue
        
        # Clip to 0-255 and normalize
        r, g, b = np.clip([r, g, b], 0, 255) / 255.0

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=(r, g, b)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, -0.3, f"{color_name}", ha='center', va='top', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    



species_name = "Swift Parrot"  
top_colors = get_top_colors(species_name, birds_df)
show_color_patches_rgb(top_colors.index, rgb_df)