import pandas as pd
import matplotlib.pyplot as plt

# Load your Excel sheet
df = pd.read_csv("~/DSP_group_project_avian_datasets/data/raw/colour/Data_S1/RGB_values_for_color_classification.csv")  
# Function to display color patches
def show_color_patches(df):
    n = len(df)
    fig, axes = plt.subplots(n, 1, figsize=(5, n*1.5))
    
    if n == 1:
        axes = [axes]  # ensure it's iterable for a single row

    for ax, (_, row) in zip(axes, df.iterrows()):
        # Extract RGB and normalize if needed (matplotlib expects 0-1)
        r = row['R'] / 255
        g = row['G'] / 255
        b = row['B'] / 255
        color = (r, g, b)
        
        # Create a rectangle with the color
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add text: color code and name
        ax.text(0.5, -0.3, f"{row['Color classification']} - {row['Colors']}", 
                ha='center', va='top', fontsize=12)

    plt.tight_layout()
    plt.show()

# Display
show_color_patches(df)