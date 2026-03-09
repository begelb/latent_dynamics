import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import os
import numpy as np

system = 'coral_hybrid3'

mfile_path = f'output/coral_hybrid3/MG/morse_sets'
model_dir = f'output/coral_hybrid3/models'
save_path = f'output/coral_hybrid3/MG/'
morse_sets = 'output/coral_hybrid3/MG/morse_sets'
scaler_dir = 'output/coral_hybrid3/data/scalers'
scaler_path = os.path.join(scaler_dir, 'scaler.gz')
encoder_path = os.path.join(model_dir, 'encoder.pt')

a0 = [0] * 13
a1 = [868.12066371, 771.75927004, 488.52361793, 340.5009617, 176.0389972, 76.92904178, 22.07863499, 12.60690058, 4.19809789, 3.14857342,
      3.14857342, 1.04847495, 1.04847495]
r = [321.84389612752153, 286.11922365736666, 181.1134685751131, 126.23608759685382, 65.26405728757342, 28.52039303466959, 8.185352800950172,
     4.673836449342548, 1.5563875376310683, 1.1672906532233014, 1.1672906532233014, 0.3887077875233593, 0.3887077875233593]
observed_state =[822, 731, 463, 323, 167, 73, 21, 12, 4, 3, 3, 1, 1] 

paper_r = [328, 291.592, 184.577736, 128.650681992, 66.512402589864, 29.065919931770566, 8.341919020418151, 4.763235760658764, 1.5861575082993686, 1.1896181312245264, 1.1896181312245264, 0.3961428376977673, 0.3961428376977673]

scaling_factor = r[0]/observed_state[0]

def U(state):
    sigma = 36
    print('slice: ', state[1:])
    total_non_recruits = sum(state[1:])
    print('total non recruits: ', total_non_recruits)
    return total_non_recruits / sigma

# scaling_factor_1 = 19/U(observed_state)
# print('scaling factor: ', scaling_factor_1)
# observed_vector = np.asarray(observed_state)
# print(U(observed_vector * scaling_factor_1))
# scaling_factor_2 = 19.5/U(observed_state)

scaling_factor_1 = 13/U(observed_state)
print('scaling factor: ', scaling_factor_1)
observed_vector = np.asarray(observed_state)
print(U(observed_vector * scaling_factor_1))
scaling_factor_2 = 25.5/U(observed_state)

new_pop_1 = (observed_vector * scaling_factor_1).tolist()
new_pop_2 = (observed_vector * scaling_factor_2).tolist()

overharvested_1 = [822, 731, 90] + [0] * 10
overharvested_2 = [822, 731, 80] + [0] * 10
overharvested_1 = [822, 731, 300] + [0] * 10
overharvested_2 = [822, 731, 30] + [0] * 10

# fixed_pts = {0: a0, 1: a1, 2: r}
fixed_pts = {0: a0, 1: a1, 2: r, 3: observed_state, 4: new_pop_1, 5: new_pop_2}
#fixed_pts = {0: a0, 1: a1, 2: r, 3: observed_state, 4: overharvested_1, 5: overharvested_2}

color_list = [ 
                '#FFB000',  # Gold 
                '#DC267F',  # Magenta
                '#648FFF', # Blue
                '#FE6100',  # Orange
                '#785EF0',  # Purple
                ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(scaler_path)
encoder = torch.load(encoder_path, map_location=device, weights_only=False)
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

def plot_single_line_segments(file_path, color_list, encoded_pts, markers=['*', '^', 's', 'd', 'p', '.']):
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
        
        plt.plot([a, b], [0, 0], color=color, linewidth=10, solid_capstyle='projecting', zorder=0)

    # --- New automation logic: Find and save extinction label ---
    extinction_latent = encoded_pts[0] # This is E(a0)
    extinction_label = None

    for _, row in df.iterrows():
        if row['a'] <= extinction_latent <= row['b']:
            extinction_label = int(row['label'])
            break

    if extinction_label is not None:
        # Save the label to a file for the next iteration to read
        label_file = os.path.join(save_path, 'extinction_label.txt')
        with open(label_file, 'w') as f:
            f.write(str(extinction_label))
        print(f"Extinction label identified as {extinction_label} and saved to {label_file}")
    # -----------------------------------------------------------
        
    labels = [f"$E(a_{0})$", f"$E(a_{1})$", f"$E(r)$", "o", 'new pop 1', 'new pop 2']
    for i, z_val in enumerate(encoded_pts):
        m_shape = markers[i]
        label = labels[i]
        plt.scatter(z_val, 0.0, 
                        marker=m_shape, 
                        color='black', 
                        s=60,     #200      # Size of the marker
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
    #plt.xlim(df[['a', 'b']].min().min(), df[['a', 'b']].max().max())
    plt.locator_params(axis='x', nbins=4) # Controls density of x-ticks
    
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

# def plot_single_line_segments2(file_path, color_list, encoded_pts, markers=['*', '.', '^', 's', 'd', 'p']):
#     """
#     Plots horizontal line segments [a, b] on a single horizontal line,
#     filtered to only show label 0 and specific encoded points.
#     """
#     # Set the global font family to serif for a LaTeX look
#     plt.rcParams.update({
#         "font.family": "serif",
#         "mathtext.fontset": "stix",
#         "font.serif": ["STIXGeneral"]
#     })

#     try:
#         df = pd.read_csv(file_path)
#         if 'a' not in df.columns:
#             df = pd.read_csv(file_path, names=['a', 'b', 'label'])
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return

#     plt.figure(figsize=(12, 2.5)) # Increased height slightly for better spacing
    
#     # 1. Plot each segment (FILTERED for label 0)
#     for _, row in df.iterrows():
#         a, b = row['a'], row['b']
#         label = int(row['label'])
        
#         if label == 0:  # Only plot Morse set 0
#             color = color_list[label % len(color_list)]
#             plt.plot([a, b], [0, 0], color=color, linewidth=36, 
#                      solid_capstyle='projecting', zorder=0)
        
#     # 2. Plot Encoded Points (FILTERED for a1 and r)
#     # labels = [E(a0), E(a1), E(r), o]
#     labels = [f"$E(a_{0})$", f"$E(a_{1})$", f"$E(r)$", "o"]
#     for i, z_val in enumerate(encoded_pts):
#         if i not in [1, 3]: # Only plot a1 (index 1) and r (index 2)
#             continue
            
#         m_shape = markers[i]
#         label = labels[i]
#         plt.scatter(z_val, 0.0, 
#                     marker=m_shape, 
#                     color='black', 
#                     s=250,           # Slightly larger marker
#                     edgecolor='black',
#                     clip_on=False, 
#                     zorder=10, 
#                     label=label)

#     # 3. Place Morse Set Text (FILTERED for label 0)
#     for label in df['label'].unique():
#         if label != 0:
#             continue
            
#         subset = df[df['label'] == label]
#         group_min = subset[['a', 'b']].min().min()
#         group_max = subset[['a', 'b']].max().max()
#         midpoint = (group_min + group_max) / 2
        
#         # Use double braces for the LaTeX exponent
#         plt.text(midpoint, 0.28, f"$|\pi^{{-1}}({int(label)})|$", 
#                  ha='center', va='bottom', 
#                  fontsize=30, color='black')
        
#     # Legend formatting
#     handles, labels_list = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels_list, handles))
#     plt.legend(
#         by_label.values(), 
#         by_label.keys(), 
#         loc='upper left',
#         fontsize=26,
#         bbox_to_anchor=(1.05, 1.3), # Adjusted anchor to bring legend closer
#         borderaxespad=0.
#     )

#     # Styling and Spines
#     plt.yticks([])
#     plt.ylim(-0.5, 0.5)
#     label_zero_data = df[df['label'] == 0]
#     if not label_zero_data.empty:
#         x_min = label_zero_data[['a', 'b']].min().min()
#         x_max = label_zero_data[['a', 'b']].max().max()
        
#     # 2. Set the x-limits based only on label 0
#     #plt.xlim(x_min, x_max)
#     plt.locator_params(axis='x', nbins=9)
    
#     ax = plt.gca()
#     for spine in ['top', 'left', 'right']:
#         ax.spines[spine].set_visible(False)
#     ax.spines['bottom'].set_position('zero')

#     ax.xaxis.set_ticks_position('bottom')
#     plt.tick_params(axis='x', which='major', labelsize=25, pad=25)
    
#     # Final adjustment to prevent cutting off the bottom/right
#     plt.subplots_adjust(right=0.75, bottom=0.3, top=0.7)
    
#     plt.savefig(os.path.join(save_path, 'morse_sets_1D_filtered.pdf'), dpi=300)
#     plt.show() 

encoded_pts = get_encoded_fixed_pts(fixed_pts, scaler, encoder)
plot_single_line_segments(morse_sets, color_list, encoded_pts)

# --- New Code to Check Fixed Points against Morse Sets ---

# 1. Load the Morse sets data if it hasn't been loaded already
# (Assuming 'df' from your script contains the Morse sets with columns 'a', 'b', and 'label')
try:
    # Ensure we are using the most recent data loaded in the script
    morse_df = pd.read_csv(mfile_path)
    if 'a' not in morse_df.columns:
        morse_df = pd.read_csv(mfile_path, names=['a', 'b', 'label'])
except Exception as e:
    print(f"Error loading Morse sets for verification: {e}")
    morse_df = None

if morse_df is not None:
    print("\n--- Fixed Point Morse Set Membership Analysis ---")
    
    # Names corresponding to the points in your fixed_pts dictionary
    # a0: Extinction, a1: Healthy Steady State, r: Repeller/Separatrix
    point_names = ['a0 (Extinction)', 'a1 (Healthy)', 'r (Repeller)']
    
    for name, z_val in zip(point_names, encoded_pts):
        found = False
        # Check against every interval in the Morse sets
        for _, row in morse_df.iterrows():
            if row['a'] <= z_val <= row['b']:
                print(f"{name}: Falls inside Morse Set [Label {int(row['label'])}] (Latent z: {z_val:.6f})")
                found = True
                break
        
        if not found:
            # If it's not in a Morse set, it's in the 'gradient-like' flow (the complement)
            print(f"{name}: Does NOT fall in any Morse set. (Latent z: {z_val:.6f})")
    print("--------------------------------------------------\n")