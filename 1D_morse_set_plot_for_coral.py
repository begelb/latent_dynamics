import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import os

mfile_path = f'output/coral/MG/morse_sets2'
model_dir = f'output/coral/models'
save_path = f'output/coral/MG/'
morse_sets = 'output/coral/MG/morse_sets2'
scaler_dir = 'output/coral/data/scalers'
scaler_path = os.path.join(scaler_dir, 'scaler.gz')
encoder_path = os.path.join(model_dir, 'encoder.pt')

a0 = [0] * 13
a1 = [868.12066371, 771.75927004, 488.52361793, 340.5009617, 176.0389972, 76.92904178, 22.07863499, 12.60690058, 4.19809789, 3.14857342,
      3.14857342, 1.04847495, 1.04847495]
r = [321.84389612752153, 286.11922365736666, 181.1134685751131, 126.23608759685382, 65.26405728757342, 28.52039303466959, 8.185352800950172,
     4.673836449342548, 1.5563875376310683, 1.1672906532233014, 1.1672906532233014, 0.3887077875233593, 0.3887077875233593]

fixed_pts = {0: a0, 1: a1, 2: r}

color_list = [ 
                '#FFB000',  # Gold 
                '#DC267F',  # Magenta
                '#648FFF', # Blue
                '#FE6100',  # Orange
                '#785EF0',  # Purple
                ]
markers = ['*', '^', 's', 'd', 'p']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(scaler_path)
encoder = torch.load(encoder_path, map_location=device)
encoder.eval()

def get_encoded_fixed_pts(fixed_pts, scaler, encoder):
    encoded_fixed_pts = []
    with torch.no_grad():
        for label, pt in fixed_pts.items():
            pt_scaled = scaler.transform([pt])
            z_tensor = encoder(torch.from_numpy(pt_scaled).float().to(device))
            z = z_tensor.cpu().numpy().flatten()[0]
            encoded_fixed_pts.append(z)
    return encoded_fixed_pts

def plot_single_line_segments(file_path, color_list, encoded_pts, markers):
    """
    Plots horizontal line segments [a, b] on a single horizontal line,
    colored by 'label'.
    """
    # Load data

    # Set the global font family to serif
    plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral"]})

    try:
        df = pd.read_csv(file_path)
        # Ensure column names match expected 'a', 'b', 'label'
        if 'a' not in df.columns:
            df = pd.read_csv(file_path, names=['a', 'b', 'label'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    plt.figure(figsize=(12, 2)) # Shorter height since it's just one line
    
    # Plot each segment at height y=0
    for _, row in df.iterrows():
        a, b = row['a'], row['b']
        label = int(row['label'])
        color = color_list[label % len(color_list)]
        
        plt.plot([a, b], [0, 0], color=color, linewidth=36, solid_capstyle='projecting', zorder=0)
        
    labels = [f"$E(a_{0})$", f"$E(a_{1})$", f"$E(r)$"]
    for i, z_val in enumerate(encoded_pts):
        m_shape = markers[i]
        label = labels[i]
        plt.scatter(z_val, 0.0, 
                        marker=m_shape, 
                        color='black', 
                        s=200,           # Size of the marker
                        edgecolor='black',
                        clip_on=False, 
                        zorder=10,        # Ensure points are drawn on top of segments
                        label=label)

    for label in df['label'].unique():
        subset = df[df['label'] == label]
        
        # Find the absolute min and max of all endpoints in this group
        group_min = subset[['a', 'b']].min().min()
        group_max = subset[['a', 'b']].max().max()
        
        # Calculate the midpoint of the entire Morse set span
        midpoint = (group_min + group_max) / 2
        
        # Place the LaTeX label above the midpoint
        plt.text(midpoint, 0.27, f"$|\pi^{{-1}}({int(label)})|$", 
                 ha='center', va='bottom', 
                 fontsize=26, color='black')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='upper left',          # The anchor point on the legend box
        fontsize=26,
        bbox_to_anchor=(1.2, 1.5),  # The point on the plot it anchors to (x=1.05 is outside)
        borderaxespad=0.1          # Padding between the axes and the border
    )

    # Formatting
    # Clean up the Y-axis since it's redundant
    plt.yticks([])
    plt.ylim(-0.5, 0.5)
    
    # Ensure x-ticks are clear
    # plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.xlim(df[['a', 'b']].min().min(), df[['a', 'b']].max().max())
    plt.locator_params(axis='x', nbins=9) # Controls density of x-ticks
    
    # Remove the box/spines except for the bottom one to make it look like a timeline
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')

    # Make sure the ticks move with the spine
    ax.xaxis.set_ticks_position('bottom')
    plt.tick_params(axis='x', which='major', labelsize=25, pad=25)
    
    plt.subplots_adjust(right=0.6, bottom=0.25, top=0.7)
    plt.savefig(os.path.join(save_path, 'morse_sets_1D.pdf'), dpi=300)
    plt.show()

encoded_pts = get_encoded_fixed_pts(fixed_pts, scaler, encoder)
plot_single_line_segments(morse_sets, color_list, encoded_pts, markers)