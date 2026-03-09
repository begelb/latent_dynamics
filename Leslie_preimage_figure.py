import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import time
import bisect
from tqdm import tqdm
import joblib
import pickle
import random
import matplotlib.colors as mcolors

ex_index = 3
mfile_path = f'output/Leslie_3D_larger_domain_tail_only1/{ex_index}/MG/morse_sets2'
model_dir = f'output/Leslie_3D_larger_domain_tail_only1/{ex_index}/models'
plot_data_dir = f'output/Leslie_3D_larger_domain_tail_only1/{ex_index}/models'
save_path = f'output/Leslie_3D_larger_domain_tail_only1/{ex_index}/'
scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/scalers'

# ex_index = 3
# mfile_path = f'output/Leslie_3D_larger_domain/{ex_index}/MG/morse_sets'
# model_dir = f'output/Leslie_3D_larger_domain/{ex_index}/models'
# plot_data_dir = f'output/Leslie_3D_larger_domain/{ex_index}/models'
# save_path = f'output/Leslie_3D_larger_domain/{ex_index}/'
# scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D_larger_domain/28.9_29.8_22.0/scalers'



scaler_path = os.path.join(scaler_dir, 'scaler.gz')

if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)
lower_bounds=[0, 0, 0]
upper_bounds=[220+1, 154+1, 108+1]
c_list = ['#ffb000', '#fe6100', '#dc267f', '#648fff', '#785ef0', '#008080']
#c_list = ['#dc267f', '#648fff']
res = 120
morse_set_data = np.loadtxt(mfile_path, delimiter=',', dtype=np.float64)
encoder_path = os.path.join(model_dir, 'encoder.pt')
dynamics_path = os.path.join(model_dir, 'dynamics.pt')

BARYCENTERS = {
    0: [
        [102.59382834, 4.62509476, 0.59276684],
        [6.47696572e-02, 7.18156798e+01, 3.23756633e+00],
        [1.20972812e+00, 4.53387600e-02, 5.02709759e+01],
        [6.60727793, 0.84680968, 0.03173713]
    ],
    1: [
        [20.09019989,  2.26201326, 21.10982997],
        [14.41254064, 14.06313992, 1.58340928],
        [43.08128567, 10.08877845, 9.84419795],
        [ 3.23144751, 30.15689997, 7.06214491]
    ]
    # 2: [
    #     [18.736549331477498, 13.115584532034248, 9.180909172423972]
    # ],
    # 3: [
    #     [0.0, 0.0, 0.0]
    # ]
}

class LeslieModel3D_Vectorized:
    def __init__(self, th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.survival_p1 = survival_p1 
        self.survival_p2 = survival_p2 

    def iterate(self, X, iterations=1):
        """
        X: NumPy array of shape (N, 3) representing the grid points
        iterations: Number of times to apply the map
        """
        curr_X = X.copy()
        
        for _ in range(iterations):
            x0 = curr_X[:, 0]
            x1 = curr_X[:, 1]
            x2 = curr_X[:, 2]
            sum_x = x0 + x1 + x2
            
            # Apply the formula to all points at once
            next_x0 = (self.th1*x0 + self.th2*x1 + self.th3*x2) * np.exp(-0.1 * sum_x)
            next_x1 = self.survival_p1 * x0
            next_x2 = self.survival_p2 * x1
            
            # Update the points for the next iteration
            curr_X = np.stack([next_x0, next_x1, next_x2], axis=1)
            
        return curr_X

# def verify_orbit(label, points, model):
#     print(f"Checking Orbit for Morse Set {label}...")
#     pts_np = np.array(points)
#     for i in range(len(pts_np)):
#         # Apply the map once
#         next_pt = model.iterate(pts_np[i:i+1], iterations=1)[0]
        
#         # Find the distance to the closest point in the original set
#         distances = np.linalg.norm(pts_np - next_pt, axis=1)
#         min_dist = np.min(distances)
#         target_idx = np.argmin(distances)
        
#         print(f"  Point {i} maps to Point {target_idx} (Dist: {min_dist:.2e})")

# # Run it
# test_model = LeslieModel3D_Vectorized(th1=28.9, th2=29.8, th3=22.0)
# for lbl, pts in BARYCENTERS.items():
#     verify_orbit(lbl, pts, test_model)

# def get_box_width_and_height(morse_set_data):
#     first_box = morse_set_data[0]
#     datapt[0], datapt[1], datapt[2], datapt[3],


# class Box:
#     def __init__(self, ID, lower_x, lower_y, upper_x, upper_y, M_label):
#         self.ID = ID
#         self.lower_x = lower_x
#         self.upper_x = upper_x
#         self.lower_y = lower_y
#         self.upper_y = upper_y
#         self.M_label = M_label
import numpy as np
import os

class Box:
    def __init__(self, ID, lower_x, lower_y, upper_x, upper_y, M_label):
        self.ID = ID
        self.lower_x = lower_x
        self.lower_y = lower_y 
        self.upper_x = upper_x 
        self.upper_y = upper_y
        self.M_label = M_label

    def get_coords(self):
        """Returns [min_x, min_y, max_x, max_y] for geometry calculations."""
        return [self.lower_x, self.lower_y, self.upper_x, self.upper_y]

    def __repr__(self):
        return f"Box(ID={self.ID}, label={self.M_label}, coords=[{self.lower_x:.3f}, {self.lower_y:.3f}, ...])"

class MorseSet:
    def __init__(self, file_path, label):
        self.label = int(label)
        self.boxes = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        
        self._load_from_file(file_path)

    def _load_from_file(self, file_path):
        try:
            # Load data: lower_x, lower_y, upper_x, upper_y, label
            data = np.loadtxt(file_path, delimiter=',', ndmin=2)
            
            # Filter for the rows matching the requested label (column index 4)
            mask = np.isclose(data[:, 4], self.label)
            filtered_data = data[mask]
            
            # Create Box objects
            # We assign a local ID (0, 1, 2...) since the file doesn't provide one
            for i, row in enumerate(filtered_data):
                # row is [lx, ly, ux, uy, label]
                new_box = Box(
                    ID=i,
                    lower_x=row[0],
                    lower_y=row[1],
                    upper_x=row[2],
                    upper_y=row[3],
                    M_label=int(row[4])
                )
                self.boxes.append(new_box)
                
        except Exception as e:
            print(f"[Error] Failed to load Morse Set {self.label}: {e}")
            self.boxes = []

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)
    
    def get_morse_set_boundary(self):
        """
        Determines the boundary edges of a Morse set (connected collection of boxes).
        
        Args:
            morse_set (MorseSet): The object containing the collection of Box objects.
            
        Returns:
            set: A set of unique Edge objects that make up the boundary.
        """
        boundary_edges = set()

        for box in self.boxes:
            # 1. Identify the 4 vertices of the current box
            # We assume the box coordinates are the grid integers (or floats aligned to grid)
            p1 = (box.lower_x, box.lower_y) # Bottom-Left
            p2 = (box.upper_x, box.lower_y) # Bottom-Right
            p3 = (box.upper_x, box.upper_y) # Top-Right
            p4 = (box.lower_x, box.upper_y) # Top-Left
            
            # 2. Create the 4 edges
            current_edges = [
                Edge(p1, p2), # Bottom
                Edge(p2, p3), # Right
                Edge(p3, p4), # Top
                Edge(p4, p1)  # Left
            ]
            
            # 3. Apply the "Toggle" Logic
            for edge in current_edges:
                if edge in boundary_edges:
                    boundary_edges.remove(edge) # It was shared, so it's internal -> Remove it
                else:
                    boundary_edges.add(edge)    # It's new -> Add it
                    
        return boundary_edges
    
class Edge:
    def __init__(self, v1, v2):
        """
        Represents an undirected edge between two grid vertices.
        v1, v2: Tuples (x, y) representing the coordinates.
        """
        # Sort vertices to ensure uniqueness: (A, B) == (B, A)
        if v1 < v2:
            self.u = tuple(v1)
            self.v = tuple(v2)
        else:
            self.u = tuple(v2)
            self.v = tuple(v1)
            
        # Determine Orientation
        # Check if Y coordinates are effectively equal (Horizontal)
        if abs(self.u[1] - self.v[1]) < 1e-9:
            self.orientation = 'horizontal'
        # Check if X coordinates are effectively equal (Vertical)
        elif abs(self.u[0] - self.v[0]) < 1e-9:
            self.orientation = 'vertical'
        else:
            # This case shouldn't happen for axis-aligned grid boxes
            self.orientation = 'diagonal'

    def __eq__(self, other):
        return isinstance(other, Edge) and self.u == other.u and self.v == other.v

    def __hash__(self):
        return hash((self.u, self.v))

    def __repr__(self):
        # Shows orientation in the printout for easier debugging
        code = 'H' if self.orientation == 'horizontal' else 'V'
        return f"Edge[{code}]({self.u} <-> {self.v})"

class Cube:
    def __init__(self, ID, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, M_label):
        self.ID = ID
        self.lower_x = lower_x
        self.lower_y = lower_y # Correct: Assigned 3rd arg to Y attribute
        self.lower_z = lower_z
        self.upper_x = upper_x # Correct: Assigned 4th arg to X attribute
        self.upper_y = upper_y
        self.upper_z = upper_z
        self.M_label = M_label

    def centroid(self):
        x = (self.lower_x + self.upper_x)/2
        y = (self.lower_y + self.upper_y)/2
        z = (self.lower_z + self.upper_z)/2
        return [x, y, z]


def sort_boxes(box_list):
    sorted_x = sorted(box_list, key=lambda box: box.lower_x)
    sorted_y = sorted(box_list, key=lambda box: box.lower_y)
    return sorted_x, sorted_y

def make_box_lists(morse_set_data):
    box_list = []
    for i, datapt in enumerate(morse_set_data):
        box_list.append(Box(i, datapt[0], datapt[1], datapt[2], datapt[3], datapt[4]))
    return sort_boxes(box_list)

def make_centroids_dict_3D(morse_set_data):
    morse_centroids_dict = dict()
    for i, datapt in enumerate(morse_set_data):
        cube = Cube(i, datapt[0], datapt[1], datapt[2], datapt[3], datapt[4], datapt[5], datapt[6])
        label = datapt[6]
        centroid = cube.centroid
        if label not in morse_centroids_dict:
            morse_centroids_dict[label] = []
        morse_centroids_dict[label].append(centroid)

    for label in morse_centroids_dict:
        morse_centroids_dict[label] = np.array(morse_centroids_dict[label])
      #  centroid_list.append(centroid)
    return morse_centroids_dict #sort_boxes(box_list)

def encode_centroids(centroid_array, encoder_path):
    scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/scalers'
    scaler_path = os.path.join(scaler_dir, 'scaler.gz')

    scaler = joblib.load(scaler_path)

    scaled_centroids = scaler.transform(centroid_array)

    centroid_tensor = torch.from_numpy(scaled_centroids).float()

    encoder = torch.load(encoder_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        latent_centroids = encoder(centroid_tensor.to(device)).cpu().numpy()

    return latent_centroids



def locate_potential_boxes(target, sorted_list, dim_prefix):
    lower_attr = f"lower_{dim_prefix}"
    upper_attr = f"upper_{dim_prefix}"
    idx = bisect.bisect_right(sorted_list, target, key=lambda b: getattr(b, lower_attr))
    candidate_indices = set()
    for i in range(idx - 1, -1, -1):
        box = sorted_list[i]
        low = getattr(box, lower_attr)
        high = getattr(box, upper_attr)
        if low <= target <= high:
            candidate_indices.add(box.ID)
        elif high < target:
            break
    return candidate_indices

lower_x_values = [0, 1, 2, 3, 4, 4.4, 4.4, 4.4, 4.4, 4.5, 5, 6]
upper_x_values = [lower_x + 0.5 for lower_x in lower_x_values]
lower_y_values = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
upper_y_values = [lower_y + 0.5 for lower_y in lower_y_values]
box_list = [Box(i, lower_x_values[i], upper_x_values[i], lower_y_values[i], upper_y_values[i], 0) for i in range(len(lower_x_values))]

# print(locate_potential_boxes(4.3, box_list, 'x'))
# print(locate_potential_boxes(1.1, box_list, 'y'))

def locate_boxes(target_x, target_y, sorted_x, sorted_y):
    potential_x = locate_potential_boxes(target_x, sorted_x, 'x')
    potential_y = locate_potential_boxes(target_y, sorted_y, 'y')
    potential_list = list(potential_x & potential_y)
    if len(potential_list) > 0:
        return potential_list[0]
    else:
        return None

# x_set = set(locate_potential_boxes(4.3, box_list, 'x'))
# y_set = set(locate_potential_boxes(1.1, box_list, 'y'))

# print(locate_boxes(4.3, 1.1, box_list, box_list))
# print('common indices: ', common_indices)

# def morse_set_locator(x, y, morse_set_data, num_att, att_only=True):
#     for datapt in morse_set_data:
#         if datapt[0] <= x <= datapt[2] and datapt[1] <= y <= datapt[3]:
#             M_label = datapt[-1]
#             print(f'({x}, {y}) is inside {datapt} with label {M_label}')
#             print('---')
#             if att_only and int(M_label) < num_att:
#                 return int(M_label)
#             if not att_only:
#                 return int(M_label)
#     return None

def morse_set_locator_efficient(x, y, morse_set_data, sorted_x_boxes, sorted_y_boxes, att_only=False, num_att=2):
    box_index = locate_boxes(x, y, sorted_x_boxes, sorted_y_boxes)
    if box_index is not None:
        M_label = morse_set_data[box_index][-1]
        print(f'({x}, {y}) is inside Box {box_index} with label {M_label}')
        print('---')
        if att_only and int(M_label) < num_att:
            return int(M_label)
        if not att_only:
            return int(M_label)
    return None

def encode_grid(lower_bounds, upper_bounds, res, encoder_path):
    grid = np.mgrid[lower_bounds[0]:upper_bounds[0]:(upper_bounds[0]-lower_bounds[0])/res, lower_bounds[1]:upper_bounds[1]:(upper_bounds[1]-lower_bounds[1])/res, lower_bounds[2]:upper_bounds[2]:(upper_bounds[2]-lower_bounds[2])/res]
    grid_points = grid.reshape(3, -1).T
   # grid_tensor = torch.from_numpy(grid_points).float()

    # 2. Determine the midpoint to split the dataset
    mid_idx = len(grid_points) // 2
    
    # 3. Initialize image_points as a copy of the original grid
    # This ensures the second half stays at 'iterations=0' by default
    image_points = grid_points.copy()

    scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D_larger_domain_tail_only/28.9_29.8_22.0/scalers'
    scaler_path = os.path.join(scaler_dir, 'scaler.gz')

    scaler = joblib.load(scaler_path)

    model = LeslieModel3D_Vectorized()
   # image_points = model.iterate(grid_points, iterations=0) # Apply 5 times


    # 5. Apply iterations ONLY to the first half
    # Replace '20' with your desired number of iterations
    image_points[:mid_idx] = model.iterate(grid_points[:mid_idx], iterations=20)

    scaled_image = scaler.transform(image_points)

    # 3. Encode the ITERATED points to get their latent positions
    # We need to send the 'image_points' through the neural network
    image_tensor = torch.from_numpy(scaled_image).float()

    encoder = torch.load(encoder_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    # with torch.no_grad():
    #     latent_points = encoder(grid_tensor)

    with torch.no_grad():
        latent_image_np = encoder(image_tensor.to(device)).cpu().numpy()


    # latent_points_np = latent_points.cpu().numpy()

    return image_points, latent_image_np

def label_3D_pts(latent_points_np, grid_points, morse_set_data):
    sorted_x_boxes, sorted_y_boxes = make_box_lists(morse_set_data)
    pt_colors = []
    x_list = []
    y_list = []
    z_list = []
    for i, pt in enumerate(latent_points_np):
        M = morse_set_locator_efficient(pt[0], pt[1], morse_set_data, sorted_x_boxes, sorted_y_boxes)
        if M is not None:
            pt_colors.append(c_list[M])
            x_list.append(grid_points[i][0])
            y_list.append(grid_points[i][1])
            z_list.append(grid_points[i][2])
    return x_list, y_list, z_list, pt_colors

def get_spatial_filter_mask(latent_points_np, morse_set_data):
    """
    Creates a boolean mask for points that fall within the global 
    extent of all Morse sets.
    """
    # 1. Determine the 'Global Bounding Box' of all Morse sets
    # morse_set_data columns: [lx, ux, ly, uy, label]
    global_lx = np.min(morse_set_data[:, 0])
    global_ly = np.min(morse_set_data[:, 1])
    global_ux = np.max(morse_set_data[:, 2])
    global_uy = np.max(morse_set_data[:, 3])
    
    print(f"Morse Domain: X=[{global_lx}, {global_ux}], Y=[{global_ly}, {global_uy}]")

    # 2. Extract latent point columns
    px = latent_points_np[:, 0]
    py = latent_points_np[:, 1]

    latent_lx = np.min(latent_points_np[:, 0])
    latent_ly = np.min(latent_points_np[:, 1])
    latent_ux = np.max(latent_points_np[:, 0])
    latent_uy = np.max(latent_points_np[:, 1])

    print(f"Latent domain: X=[{latent_lx}, {latent_ux}, Y=[{latent_ly}, {latent_uy}]]")

    # 3. Create a mask of points that are within this global box
    spatial_mask = (px >= global_lx) & (px <= global_ux) & \
                   (py >= global_ly) & (py <= global_uy)
    
    return spatial_mask

def label_3D_pts_vectorized(latent_points_np, grid_points, morse_set_data):
    # Initialize arrays
    num_pts = latent_points_np.shape[0]
    # Use -1 or a specific value to indicate "no box found"
    assigned_labels = np.full(num_pts, -1, dtype=int)
    
    # Extract coordinates once
    px = latent_points_np[:, 0]
    py = latent_points_np[:, 1]
    
    # Iterate through boxes instead of points
    for datapt in tqdm(morse_set_data, desc="Processing Morse Boxes"):
        lx, ly, ux, uy, label = datapt
        # Create a boolean mask of all points inside this box
        mask = (px >= lx) & (px <= ux) & (py >= ly) & (py <= uy)
        assigned_labels[mask] = int(label)
    
    # Filter only points that were assigned a label
    valid_mask_att_only = (assigned_labels >= 0) #& (assigned_labels < 2)
  #  valid_mask = assigned_labels != -1
    x_list = grid_points[valid_mask_att_only, 0]
    y_list = grid_points[valid_mask_att_only, 1]
    z_list = grid_points[valid_mask_att_only, 2]
    pt_colors = [c_list[lbl] for lbl in assigned_labels[valid_mask_att_only]]
    print(set(pt_colors))
    
    return x_list, y_list, z_list, pt_colors

def label_3D_pts_with_filter(latent_points_np, grid_points, morse_set_data):
    # Step A: Filter out points that aren't even near any Morse set
    spatial_mask = get_spatial_filter_mask(latent_points_np, morse_set_data)
    
    # Prune the data immediately
    filtered_latent = latent_points_np[spatial_mask]
    filtered_grid = grid_points[spatial_mask]
    
    print(f"Points remaining after spatial filtering: {len(filtered_latent)} / {len(latent_points_np)}", flush=True)

    # Step B: Run your vectorized box-check on the remaining points only
    # (Using the vectorized function we discussed previously)
    return label_3D_pts_vectorized(filtered_latent, filtered_grid, morse_set_data)

def get_3d_trajectories(start_points, num_steps=1):
    """
    Takes a set of 3D points and computes their images under the map 
    for num_steps. Returns a list of trajectories.
    """
    model = LeslieModel3D_Vectorized()
    # trajectories shape: (num_points, num_steps + 1, 3)
    trajectories = np.zeros((len(start_points), num_steps + 1, 3))
    trajectories[:, 0, :] = start_points
    
    current_points = start_points
    for s in range(1, num_steps + 1):
        current_points = model.iterate(current_points, iterations=1)
        trajectories[:, s, :] = current_points
        
    return trajectories

def add_vector_field_arrows(ax, x_arr, y_arr, z_arr, colors_arr, featured_color='#000000', num_steps=1):
    """
    Draws arrows from points of featured_color to their images under the map.
    Optimized for performance using vectorized plotting calls.
    """
    # 1. Identify the points that match the featured color
    indices = np.where(colors_arr == featured_color)[0]
    if len(indices) == 0:
        return
    
    # Extract starting coordinates
    start_points = np.column_stack((x_arr[indices], y_arr[indices], z_arr[indices]))
    
    # 2. Get the 3D trajectories (Shape: N x (Steps+1) x 3)
    # This assumes you have the get_3d_trajectories function defined from previous steps
    trajectories = get_3d_trajectories(start_points, num_steps=num_steps)
    
    # 3. Vectorize the data for plotting
    # We want to plot segments: Step 0->1, Step 1->2, ...
    
    # "Starts" of arrows are indices 0 to N-1
    # "Ends" of arrows are indices 1 to N
    p_starts = trajectories[:, :-1, :]  # Shape: (N, Steps, 3)
    p_ends   = trajectories[:, 1:, :]   # Shape: (N, Steps, 3)
    
    # Flatten the arrays to put all steps and all points into one long list
    # Shape becomes: (N*Steps, 3)
    flat_starts = p_starts.reshape(-1, 3)
    flat_ends   = p_ends.reshape(-1, 3)
    
    # 4. Project to 2D (Using indices 0 and 2 as per your specific projection)
    X_start = flat_starts[:, 0]
    Y_start = flat_starts[:, 2] # Using Z as the Y-axis on the plot
    
    X_end   = flat_ends[:, 0]
    Y_end   = flat_ends[:, 2]
    
    # Calculate vector components (U, V)
    U = X_end - X_start
    V = Y_end - Y_start
    
    # 5. Single Draw Calls (Massively faster)
    
    # Plot all destination points at once
    ax.scatter(X_end, Y_end, color=featured_color, s=10, alpha=0.6)
    
    # Plot all arrows at once
    ax.quiver(
        X_start, Y_start,  # Tail positions
        U, V,              # Vector directions
        angles='xy', scale_units='xy', scale=1, 
        color='black',
        width=0.004,
        headwidth=3,
        alpha=0.4
    )
def add_single_point_trajectory(ax, x_arr, y_arr, z_arr, colors_arr, featured_color='#000000', num_steps=1, skip_first_n=0):
    """
    Finds the FIRST point of the featured_color and plots its trajectory for num_steps.
    Vectorized for performance (fast even with many steps).
    """
    # 1. Identify the points that match the featured color
    indices = np.where(colors_arr == featured_color)[0]
    if len(indices) == 0:
        print(f"No points found with color {featured_color}")
        return
    
    # --- CHANGE: Select ONLY the first point found ---
    idx = indices[0]
    start_point = np.array([[x_arr[idx], y_arr[idx], z_arr[idx]]])

    if skip_first_n > 0:
        model = LeslieModel3D_Vectorized()
        # Iterate forward silently without recording trajectory
        start_point = model.iterate(start_point, iterations=skip_first_n)
    
    # 2. Get the 3D trajectory (Shape: 1 x (Steps+1) x 3)
    trajectory = get_3d_trajectories(start_point, num_steps=num_steps)
    
    # Extract the single path (Shape: Steps+1 x 3)
    path = trajectory[0]
    
    # 3. Project to 2D (Using X and Z axes as per your previous code)
    X = path[:, 0]
    Y = path[:, 2]
    
    # 4. Create Vector Segments (Start -> End)
    X_start = X[:-1]
    Y_start = Y[:-1]
    X_end   = X[1:]
    Y_end   = Y[1:]
    
    U = X_end - X_start
    V = Y_end - Y_start
    
    # 5. Plot everything in ONE call (Fast!)
    # Plot the destination points
    ax.scatter(X_end, Y_end, color='black', s=20, alpha=0.8, zorder=30)
    
    # Plot the arrow trail
    ax.quiver(
        X_start, Y_start, 
        U, V, 
        angles='xy', scale_units='xy', scale=1, 
        color='black',
        width=0.005, headwidth=2, alpha=0.6, zorder=29
    )

def morse_preimage_plot(x_list, y_list, z_list, pt_colors, barycenters, save_path, featured_color):
    start_time = time.perf_counter()
    fig = plt.figure(figsize=(10, 7))
   # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    max_points = 100      # Total points to show per color
    keep_last_n = 20     # Specifically preserve the last N points added

    # 1. Convert to arrays for indexing
    x_arr, y_arr, z_arr = np.array(x_list), np.array(y_list), np.array(z_list)
    colors_arr = np.array(pt_colors)

    unique_colors = np.unique(colors_arr)
    final_indices = []

    for color in unique_colors:
        color_indices = np.where(colors_arr == color)[0]
        num_pts = len(color_indices)
        
        if num_pts <= max_points:
            # Keep everything if we are under the budget
            sampled_indices = color_indices
            print('block 1')
        elif color_dict[color] == 1:
            # Keep everything if we are under the budget
            sampled_indices = color_indices
            random_sample = np.random.choice(sampled_indices, max_points, replace=False)
            print('block 2')
        else:
            print('block 3')
            # 1. Grab the last N indices (the most recent points)
            last_indices = color_indices[-keep_last_n:]
            
            # 2. Identify the pool for the random sample (everything EXCEPT the last N)
            remaining_pool = color_indices[:-keep_last_n]
            
            # 3. Calculate how many more we need to hit the max_points budget
            num_to_sample = max_points - keep_last_n
            
            # 4. Randomly sample from the pool
            random_sample = np.random.choice(remaining_pool, num_to_sample, replace=False)
            
            # Combine the guaranteed recent points with the random background points
            sampled_indices = np.concatenate([random_sample, last_indices])
        
        final_indices.extend(sampled_indices)

    # Use these indices for plotting
    x_final = x_arr[final_indices]
    y_final = y_arr[final_indices]
    z_final = z_arr[final_indices]
    colors_final = colors_arr[final_indices]

  #  add_vector_field_arrows(ax, x_final, y_final, z_final, colors_final, 
  #                     featured_color=featured_color, num_steps=1000)
    #add_single_point_trajectory(ax, x_arr, y_arr, z_arr, colors_arr, featured_color=featured_color, num_steps=20, skip_first_n=0)


    # 5. Plot the subsampled data
  #  img = ax.scatter(x_final, y_final, z_final, c=colors_final, s=30, alpha=0.9)
    img = ax.scatter(x_final, z_final, c=colors_final, s=30, alpha=0.9)

    # img = ax.scatter(x_list, y_list, z_list, c=pt_colors, s=30, alpha=0.5)

    markers = ['*', '^', 'D', 's', 'p']
    
    # Track handles for a separate barycenter legend if desired
    bary_legend_elements = []

    # Iterate through the barycenters dictionary
    # for label, points in barycenters.items():
    #     if int(label) < len(markers):
    #         marker_shape = markers[int(label)]
            
    #         # Convert list of points to a numpy array for easy slicing
    #         pts_array = np.array(points)
            
    #         # Plot the barycenters as large black markers
    #         ax.scatter(pts_array[:, 0], pts_array[:, 1], pts_array[:, 2], 
    #                    color='black', 
    #                    marker=marker_shape, 
    #                    s=150,           # Large size
    #                    edgecolors='black', 
    #                    linewidths=1.5,
    #                    label=f'Barycenter {label}',
    #                    alpha=0.5)       # Ensure they are drawn on top of the grid
            
    #         # Add to legend elements
    #         bary_legend_elements.append(
    #             Line2D([0], [0], marker=marker_shape, color='w', 
    #                    label=f'Barycenter {label}',
    #                    markerfacecolor='black', markersize=12)
    #         )

    categories = ["0", "1", "2", "3", "4", "5"]

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=categories[i],
            markerfacecolor=c_list[i], markersize=10)
        for i in range(len(categories))
    ]

    #ax.legend(handles=legend_elements + bary_legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.legend(handles=legend_elements + bary_legend_elements, loc='upper right')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_3$')
   # ax.set_zlabel(r'$x_3$')
    end_time = time.perf_counter()
    duration_mins = round((end_time - start_time)//60)
    print(f"Time to produce plot: {duration_mins} minutes")

    plt.savefig(os.path.join(save_path, 'MG', 'e'), dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()

def find_and_save_preimage_samples(target_k, num_samples=100, iterations=21, batch_size=10):
    """
    Randomly samples 3D points until 100 points are found that land in Morse set k.
    Saves the collection of points as a single NumPy array in a pickle file.
    """
    # 1. Setup components from global script variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.eval()
    model = LeslieModel3D_Vectorized()
    
    found_points = []
    attempt = 0

    print(f"Searching for {num_samples} points in the preimage of Morse set {target_k}...")

    lower_bounds = [0, 0, 0]
    upper_bounds = [110, 80, 55]
    print('Using upper bounds: ', upper_bounds)
    while len(found_points) < num_samples:
        attempt += 1
        # 2. Sample a batch of 3D seeds
        seeds = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(batch_size, 3))
        
        # 3. Iterate the seeds forward
      #  iterations = random.randint(10, 100)
        # iterations = 0
        # print('using iterations : ', iterations%4)
        iterated_seeds = model.iterate(seeds, iterations=iterations)
        
        # 4. Scale and Encode to latent space
        scaled_iterated = scaler.transform(iterated_seeds)
        with torch.no_grad():
            latent_z = encoder(torch.from_numpy(scaled_iterated).float().to(device)).cpu().numpy()
            
        px, py = latent_z[:, 0], latent_z[:, 1]
        
        # 5. Filter for target Morse set boxes
        target_boxes = morse_set_data[morse_set_data[:, 4] == target_k]
        
        # Check this batch against all boxes for Morse set k
        batch_matches = np.zeros(batch_size, dtype=bool)
        for box in target_boxes:
            lx, ly, ux, uy = box[0], box[1], box[2], box[3]
            mask = (px >= lx) & (px <= ux) & (py >= ly) & (py <= uy)
            batch_matches |= mask
        
        # Collect matches from this batch
        if np.any(batch_matches):
            matches = iterated_seeds[batch_matches]
            found_points.extend(matches)
            print(f"  Found {len(matches)} matches in batch {attempt}. Total: {min(len(found_points), num_samples)}/{num_samples}")

        if attempt % 10 == 0 and len(found_points) == 0:
            print(f"  Still searching... Processed {attempt * batch_size} points without a match.")

    # 6. Finalize and Save
    # Trim to exactly the number requested if the last batch overshot
    final_points = np.array(found_points[:num_samples])
    
    save_file = os.path.join(save_path, f'preimage_samples_k{target_k}_{num_samples}pts_v2.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(final_points, f)
        
    print(f"Success! {num_samples} points saved to: {save_file}")
    return final_points

def get_max_dynamics_loss(target_color, x_list, y_list, z_list, pt_colors):
    """
    Computes the Latent Dynamics Consistency Loss (L3) for all points of a specific color
    and returns the maximum loss value found.
    
    L3 = || Encoder(True_Next_Step) - Dynamics(Encoder(Current_Step)) ||^2
    """
    # 1. Identify the points that match the target color
    colors_arr = np.array(pt_colors)
    indices = np.where(colors_arr == target_color)[0]
    
    if len(indices) == 0:
        print(f"No points found with color {target_color}")
        return 0.0
    
    # 2. Extract physical 3D coordinates (Current Step x)
    # Convert lists to arrays if they aren't already
    x_arr, y_arr, z_arr = np.array(x_list), np.array(y_list), np.array(z_list)
    current_pts = np.column_stack((x_arr[indices], y_arr[indices], z_arr[indices]))
    
    # 3. Compute True Next Step (Physical Evolution y = F(x))
    # We use the Leslie model to find where these points actually go in 3D
    model_phys = LeslieModel3D_Vectorized()
    true_next_pts = model_phys.iterate(current_pts, iterations=1)
    
    # 4. Load Scaler (Ensure scaler_path is defined in your script scope)
    # Using the global scaler_path variable from your script
    scaler = joblib.load(scaler_path)
    
    # Scale both sets of points
    current_scaled = scaler.transform(current_pts)
    true_next_scaled = scaler.transform(true_next_pts)
    
    # 5. Load Models (Ensure encoder/dynamics are loaded/on device)
    # Using global encoder/dynamics variables or reloading if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure models are in eval mode
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.eval()
    dynamics = torch.load(dynamics_path, map_location=device, weights_only=False)
    dynamics.eval()
    
    # 6. Compute Latent Representations
    with torch.no_grad():
        # A. Encode current points -> z_t
        t_current = torch.from_numpy(current_scaled).float().to(device)
        z_current = encoder(t_current)
        
        # B. Encode true next points -> z_{t+1} (Ground Truth in Latent Space)
        t_next_true = torch.from_numpy(true_next_scaled).float().to(device)
        z_next_true_encoded = encoder(t_next_true)
        
        # C. Predict next latent state -> z_{pred} (Learned Dynamics)
        z_next_pred = dynamics(z_current)
        
        # 7. Compute the Squared Euclidean Error (Loss) per point
        # Loss = Sum((z_true - z_pred)^2) across dimensions
        diff = z_next_true_encoded - z_next_pred
        # Sum of squares per row (dim=1)
        loss_per_point = torch.sum(diff ** 2, dim=1).cpu().numpy()
        
    # 8. Find Max
    max_loss = np.max(loss_per_point)
    mean_loss = np.mean(loss_per_point)
    
    print(f"--- Dynamics Loss (L3) Analysis for {target_color} ---")
    print(f"Points analyzed: {len(indices)}")
    print(f"Max Loss: {max_loss:.6f}")
    print(f"Mean Loss: {mean_loss:.6f}")
    
    return max_loss

import numpy as np

# ==========================================
# 1. HELPER: Check if point is "In Range"
# ==========================================
def is_in_range(point, edge):
    """
    Determines if the point 'projects' onto the edge segment.
    
    Args:
        point (tuple): (x, y) coordinates.
        edge (Edge): An Edge object with attributes u, v, and orientation.
        
    Returns:
        bool: True if point is between endpoints, False otherwise.
    """
    px, py = point
    (ux, uy) = edge.u
    (vx, vy) = edge.v
    
    # Epsilon for float comparisons
    eps = 1e-9

    if edge.orientation == 'horizontal':
        # Check x-coordinate range: min_x <= px <= max_x
        min_x = min(ux, vx)
        max_x = max(ux, vx)
        return (min_x - eps <= px <= max_x + eps)

    elif edge.orientation == 'vertical':
        # Check y-coordinate range: min_y <= py <= max_y
        min_y = min(uy, vy)
        max_y = max(uy, vy)
        return (min_y - eps <= py <= max_y + eps)
    
    return False

# ==========================================
# 2. HELPER: Orthogonal Distance
# ==========================================
def get_orthogonal_distance(point, edge):
    """
    Calculates orthogonal distance from point to the line containing the edge.
    Note: This assumes the point is 'in range' or that we only care about the
    axis-aligned difference.
    """
    px, py = point
    (ux, uy) = edge.u
    
    if edge.orientation == 'horizontal':
        # Distance in Y (Edge y is constant, so use uy)
        return abs(py - uy)
    
    elif edge.orientation == 'vertical':
        # Distance in X (Edge x is constant, so use ux)
        return abs(px - ux)
    
    return float('inf')

# ==========================================
# 3. Distance to Boundary of N
# ==========================================
def distance_point_to_boundary(point, boundary_edges):
    """
    Computes the minimum distance from point to any 'in range' boundary edge.
    
    Args:
        point (tuple): (x, y)
        boundary_edges (set/list): Collection of Edge objects.
        
    Returns:
        float: The minimum distance. Returns infinity if no edges are in range.
    """
    min_dist = float('inf')
    found_in_range = False
    
    for edge in boundary_edges:
        if is_in_range(point, edge):
            found_in_range = True
            dist = get_orthogonal_distance(point, edge)
            if dist < min_dist:
                min_dist = dist
                
    return min_dist

# ==========================================
# 4. Final Computation: Min Boundary Separation
# ==========================================
import torch
import numpy as np

def compute_min_boundary_separation(morse_set, dynamics_model, device):
    """
    Computes the minimum distance from G(Vertices_of_N) to the boundary of N,
    where G is a PyTorch model.
    
    Args:
        morse_set (MorseSet): The set containing boxes.
        dynamics_model (torch.nn.Module): The loaded PyTorch model G.
        device (torch.device): 'cpu' or 'cuda'.
        
    Returns:
        float: The minimum distance found.
    """
    # 1. Extract Boundary Edges
    boundary_edges = morse_set.get_morse_set_boundary()
    
    # 2. Collect all unique vertices of N
    unique_vertices = set()
    for box in morse_set.boxes:
        unique_vertices.add((box.lower_x, box.lower_y))
        unique_vertices.add((box.upper_x, box.lower_y))
        unique_vertices.add((box.upper_x, box.upper_y))
        unique_vertices.add((box.lower_x, box.upper_y))
    
    if not unique_vertices:
        print("[Warning] Morse set has no vertices.")
        return 0.0

    # Convert to NumPy array for batch processing
    # Shape: (Num_Vertices, 2)
    vertices_arr = np.array(list(unique_vertices), dtype=np.float32)
    
    # 3. Apply Latent Dynamics G (Batch Inference)
    dynamics_model.eval() # Ensure model is in evaluation mode
    
    with torch.no_grad():
        # Convert to Tensor on Device
        inputs = torch.from_numpy(vertices_arr).to(device)
        
        # Forward Pass
        # Result shape: (Num_Vertices, 2)
        outputs = dynamics_model(inputs)
        
        # Move back to CPU and NumPy
        mapped_vertices = outputs.cpu().numpy()

    # 4. Compute Minimum Distance
    global_min_distance = float('inf')
    valid_points_count = 0
    
    for v_mapped in mapped_vertices:
        # v_mapped is a numpy array [x, y], convert to tuple for helper
        point = tuple(v_mapped)
        
        d = distance_point_to_boundary(point, boundary_edges)
        
        if d < global_min_distance:
            global_min_distance = d
        
        if d != float('inf'):
            valid_points_count += 1

    print(f"Evaluated {len(vertices_arr)} vertices. {valid_points_count} fell within orthogonal range of boundary.")
    
    # If no points were in range (all drifted diagonally away), return infinity or handle gracefully
    return global_min_distance

# --- Example Usage ---
# max_val = get_max_dynamics_loss('#ffb000', x_list, y_list, z_list, pt_colors)

# Example call:
# find_and_save_preimage_samples(target_k=0, num_samples=500, iterations=0, batch_size=100000)

# Example usage:
#find_and_save_preimage_samples(target_k=4, num_samples=20, iterations=20, batch_size=100)

image_points, latent_points_np = encode_grid(lower_bounds, upper_bounds, res, encoder_path)
x_list, y_list, z_list, pt_colors = label_3D_pts_with_filter(latent_points_np, image_points, morse_set_data)
plot_data = {
    'x': x_list,
    'y': y_list,
    'z': z_list,
    'colors': pt_colors
}

with open(os.path.join(plot_data_dir, 'preimage_plot_data2.pkl'), 'wb') as f:
    pickle.dump(plot_data, f)

# 3. To load it back later:
with open(os.path.join(plot_data_dir, 'preimage_plot_data2.pkl'), 'rb') as f:
    data = pickle.load(f)
    x_list, y_list, z_list, pt_colors = data['x'], data['y'], data['z'], data['colors']

# with open('output/Leslie_3D_larger_domain_tail_only1/3/preimage_samples_k4_20pts.pkl', 'rb') as f:
#     new_samples = pickle.load(f)

# with open('output/Leslie_3D_larger_domain_tail_only1/3/preimage_samples_k0_100pts_v2.pkl', 'rb') as f:
#     new_samples0 = pickle.load(f)

colors_arr = np.array(pt_colors)
print(len(colors_arr))
print(set(colors_arr))

c_list = ['#ffb000', '#fe6100', '#dc267f', '#648fff', '#785ef0', '#008080']
color_dict = {}
for i, color in enumerate(c_list):
    color_dict[color] = i

M_to_color_dict = {}
for M in range(len(c_list)):
    M_to_color_dict[M] = c_list[M]

print('C: ', len(set(pt_colors)))
for c in set(pt_colors):
    print(color_dict[c])

# --- 1. Find the preimage points ---
# Target k=0 (or whichever set you are interested in)
target_k = 4
num_preimage_pts = 20

# Call the function (returns a numpy array of shape (100, 3))
# new_samples = find_and_save_preimage_samples(target_k=target_k, num_samples=num_preimage_pts)

# # --- 2. Add these samples to your existing plot lists ---
# # Assuming x_list, y_list, z_list, and pt_colors are already loaded
# x_list = np.concatenate([x_list, new_samples[:, 0]])
# y_list = np.concatenate([y_list, new_samples[:, 1]])
# z_list = np.concatenate([z_list, new_samples[:, 2]])
# # Add the specific color for these points to the color list
# pt_colors.extend([M_to_color_dict[target_k]] * num_preimage_pts)

# x_list = np.concatenate([x_list, new_samples0[:, 0]])
# y_list = np.concatenate([y_list, new_samples0[:, 1]])
# z_list = np.concatenate([z_list, new_samples0[:, 2]])
# # Add the specific color for these points to the color list
# pt_colors.extend([M_to_color_dict[0]] * 100)

m_label = 0
target_color = M_to_color_dict[m_label]
max_semi_conj_error = get_max_dynamics_loss(target_color, x_list, y_list, z_list, pt_colors)

M = MorseSet(mfile_path, m_label)
# Extract boundary
boundary = M.get_morse_set_boundary()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dynamics_model = torch.load(dynamics_path, map_location=device, weights_only=False)

tau = compute_min_boundary_separation(M, dynamics_model, device)
print('tau: ', tau)
print('Semiconjugacy error exceeds tolerance: ', max_semi_conj_error > tau)
if max_semi_conj_error > tau:
    print('ATTRACTING BLOCK IS SPURIOUS')
    print('* confetti *')
elif tau < max_semi_conj_error:
    print('Theorem numerically satisfied for morse node (WARNING: Have not used Lipschitz constant) ', m_label)



# print(f"Found {len(boundary)} boundary edges.")

# # Optional: Visualization check
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(8, 8))

# # Draw the boundary edges
# for edge in boundary:
#     (x1, y1) = edge.u
#     (x2, y2) = edge.v
#     ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

# # Set limits based on one of the boxes to see it
# first_box = M.boxes[0]
# pad = 1.0
# ax.set_xlim(first_box.lower_x - pad, first_box.upper_x + pad)
# ax.set_ylim(first_box.lower_y - pad, first_box.upper_y + pad)
# plt.title(f"Boundary of Morse Set {M.label}")
# plt.axis('equal')
# plt.show()


# --- 3. Update the Plotting Function ---
# Call your existing plot function with the updated lists

#morse_preimage_plot(x_list, y_list, z_list, pt_colors, barycenters=BARYCENTERS, save_path=save_path, featured_color=M_to_color_dict[4])

morse_preimage_plot(x_list, y_list, z_list, pt_colors, barycenters=BARYCENTERS, save_path=save_path, featured_color=target_color)


# Define the scaler path globally (moved out of encode_grid if necessary)
color_list = ['#ffb000', '#dc267f', '#fe6100', '#648fff', '#785ef0', '#008080', '#fcc2e8']


def plot_latent_trajectory_small_pts(morse_set_data, barycenters, encoder_path, dynamics_path, 
                                     scaler_path, save_filename, color_list=None, trajectory_steps=4):
    """
    Plots colored Morse sets with trajectory points.
    If trajectory_steps < 5: All markers/arrows use uniform styling (Black, fixed size).
    If trajectory_steps >= 5: Markers grow and fade from gray to black.
    """
    if color_list is None:
        color_list = ['#ffb000', '#fe6100', '#dc267f', '#648fff', '#785ef0', '#008080']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    dynamics = torch.load(dynamics_path, map_location=device, weights_only=False)
    encoder.eval(); dynamics.eval()

    # 1. Background: Morse Sets
    lx, ly, ux, uy = morse_set_data[:, 0], morse_set_data[:, 1], morse_set_data[:, 2], morse_set_data[:, 3]
    labels = morse_set_data[:, 4].astype(int)
    centers_x, centers_y = (lx + ux) / 2, (ly + uy) / 2
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = (labels == lbl)
        color = color_list[lbl % len(color_list)]
        
        plt.scatter(centers_x[mask], centers_y[mask], color=color, marker='s', s=12, 
                    alpha=1, edgecolors='none', label=f'Morse set {lbl}', zorder=1)

    # 2. Trajectories
    markers = ['s', '*', 'D', '*', '^', 'p']
    gray_to_black = mcolors.LinearSegmentedColormap.from_list("gb", ["#cccccc", "#000000"])
    
    # --- CHECK FOR SHORT TRAJECTORY ---
    is_short = trajectory_steps < 5

    with torch.no_grad():
        for label, points in barycenters.items():
            m_shape = markers[int(label) % len(markers)]
            
            # Using only the first point as requested in your snippet
            start_pt = points[0]
            
            pt_scaled = scaler.transform([start_pt])
            z = encoder(torch.from_numpy(pt_scaled).float().to(device))
            
            trajectory = [z.cpu().numpy()[0]]
            for i in range(trajectory_steps):
                z = dynamics(z)
                trajectory.append(z.cpu().numpy()[0])
            
            traj_np = np.array(trajectory)
            
            # 1. Plot the connecting line
            line_alpha = 0.3 if is_short else 0.1
            plt.plot(traj_np[:, 0], traj_np[:, 1], color='black', alpha=line_alpha, 
                    linestyle='-', linewidth=0.8, zorder=5)

            # 2. Plot the points and arrows
            for i in range(len(traj_np)):
                prog = i / (len(traj_np) - 1) if len(traj_np) > 1 else 1.0
                
                # --- STYLING LOGIC ---
                if is_short:
                    # Uniform Styling for short trajectories
                    size = 20
                    current_color = 'black'
                    lw = 1.0
                    arrow_alpha = 1
                else:
                    # Progressive Styling for long trajectories
                    size = 25 + (prog * 45) 
                    current_color = gray_to_black(prog)
                    lw = 0.5 + (prog * 0.7)
                    arrow_alpha = 0.2 + (prog * 0.4)
                
                plt.scatter(traj_np[i, 0], traj_np[i, 1], 
                            facecolor=current_color, 
                            marker=m_shape, 
                            s=size, 
                            edgecolors='black', 
                            linewidths=lw, 
                            zorder=10 + i)

                # Directional arrows
                if i < len(traj_np) - 1:
                    plt.annotate('', xy=(traj_np[i+1, 0], traj_np[i+1, 1]), 
                                xytext=(traj_np[i, 0], traj_np[i, 1]),
                                arrowprops=dict(arrowstyle='-|>', color='black', 
                                            lw=0.8, alpha=arrow_alpha, 
                                            mutation_scale=10),
                                zorder=100)

            # 3. Add Legend Entry (Dummy point)
            # plt.scatter([], [], color='black', marker=m_shape, s=70, 
            #             label=f'Attractor {label}', edgecolors='black')

    # 3. Formatting
    plt.xlabel('$z_1$', fontsize=16); plt.ylabel('$z_2$', fontsize=16)
    #plt.title('Latent Dynamics', fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # leg = plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=16)
    
    # for handle, text_obj in zip(leg.legend_handles, leg.get_texts()):
    #     label_text = text_obj.get_text()
    #     handle.set_alpha(1.0)
        
    #     if "Attractor 0" in label_text or "Attractor 1" in label_text:
    #         handle.set_sizes([50.0]) 
    #     elif "Morse set" in label_text:
    #         handle.set_sizes([80.0])
    #     else:
    #         handle.set_sizes([110.0])
            
    # plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    plt.show()

# # --- Run the updated plot ---
#     '''' Here is second plot function'''
# plot_latent_trajectory_small_pts(
#     morse_set_data, BARYCENTERS, encoder_path, dynamics_path, scaler_path, 
#     os.path.join(save_path, "latent_trajectory.png"), color_list=color_list, trajectory_steps=4
# )


# def get_sampled_encoded_3d_centroids(true_mfile_path, encoder_path, samples_per_label=100):
#     """
#     Loads 3D Morse sets, samples 100 centroids per label, and encodes them.
#     Uses existing make_centroids_dict_3D and encode_centroids functions.
#     """
#     # 1. Load the 3D data
#     true_data_3d = np.loadtxt(true_mfile_path, delimiter=',', dtype=np.float64)
    
#     # 2. Get the centroids dictionary using your existing function
#     # Note: This returns {label: [centroid_method, ...]}
#     centroids_dict = make_centroids_dict_3D(true_data_3d)
    
#     processed_latent_data = [] # List of (latent_pts, label)
    
#     for label, methods in centroids_dict.items():
#         # Resolve method references to actual [x, y, z] coordinates
#         coords = np.array([m() if callable(m) else m for m in methods])
        
#         # 3. Subsample 100 points for this specific label
#         if len(coords) > samples_per_label:
#             indices = np.random.choice(len(coords), samples_per_label, replace=False)
#             coords = coords[indices]
            
#         # 4. Encode using your existing function
#         latent_pts = encode_centroids(coords, encoder_path)
#         processed_latent_data.append((latent_pts, label))
        
#     return processed_latent_data

# def plot_latent_comparison_sampled(morse_set_data, sampled_3d_info, save_filename):
#     """
#     Plots ALL 2D Morse sets and overlays SAMPLED encoded 3D centroids.
#     """
#     # User specified color list
#     color_list = ['#ffb000', '#dc267f', '#fe6100', '#648fff', '#785ef0', '#12973c', '#fcc2e8']
#     # Markers to distinguish 3D labels
#     markers_3d = ['o', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', 'H']

#     plt.figure(figsize=(10, 8))
    
#     # 1. Plot ALL 2D Morse sets (Background)
#     lx, ly, ux, uy = morse_set_data[:, 0], morse_set_data[:, 1], morse_set_data[:, 2], morse_set_data[:, 3]
#     labels_2d = morse_set_data[:, 4].astype(int)
#     centers_x, centers_y = (lx + ux) / 2, (ly + uy) / 2
    
#     for lbl in np.unique(labels_2d):
#         mask = (labels_2d == lbl)
#         plt.scatter(centers_x[mask], centers_y[mask], 
#                     color=color_list[lbl % len(color_list)], 
#                     marker='s', s=12, alpha=0.4, edgecolors='none', 
#                     label=f'2D Morse Set {lbl}', zorder=1)

#     # 2. Plot SAMPLED 3D Centroids (Overlay)
#     for i, (pts, label_3d) in enumerate(sampled_3d_info):
#         m_style = markers_3d[i % len(markers_3d)]

#         plt.scatter(pts[:, 0], pts[:, 1], 
#                     color='black', 
#                     marker=m_style, 
#                     s=30,             # Smaller size
#                     edgecolors='white',  # No white border
#                     zorder=10, 
#                     label=f'3D Centroids (Label {int(label_3d)})')

#     plt.xlabel('$z_1$', fontsize=12)
#     plt.ylabel('$z_2$', fontsize=12)
#     plt.title('Latent Space: 2D Morse Sets vs. Subsampled 3D Dynamics', fontsize=13)
#     plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
#     plt.grid(True, linestyle='--', alpha=0.2)
#     plt.tight_layout()
    
#     plt.savefig(save_filename, dpi=300)
#     plt.show()

# # --- Integration Logic ---
# true_3d_file = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/true_dynamics/2/morse_sets'

# # 1. Process and sample the 3D data
# # This uses your existing 'encoder_path' and your existing functions
# # sampled_3d_info = get_sampled_encoded_3d_centroids(true_3d_file, encoder_path, samples_per_label=25)

# # 2. Create the final plot
# # plot_latent_comparison_sampled(
# #     morse_set_data=morse_set_data, 
# #     sampled_3d_info=sampled_3d_info, 
# #     save_filename=os.path.join(save_path, "latent_comparison_sampled_3d.png")
# # )