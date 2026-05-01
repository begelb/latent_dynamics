import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import joblib
import pickle
import matplotlib.colors as mcolors
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mfile_path = f'output/Leslie_3D/spurious_attractor_ex/MG/morse_sets'
model_dir = f'output/Leslie_3D/spurious_attractor_ex/models'
plot_data_dir = f'output/Leslie_3D/spurious_attractor_ex/plot_data'
save_path = f'output/Leslie_3D/spurious_attractor_ex'
scaler_dir = '/Users/brittany/Documents/GitHub/PCA-Leslie/output/Leslie_3D/28.9_29.8_22.0/scalers'

color_list = ['#ffb000', '#dc267f', '#fe6100', '#648fff', '#785ef0', '#008080', '#fcc2e8']

# m_label for tolerance / semiconjugacy error analysis
m_label = 0

scaler_path = os.path.join(scaler_dir, 'scaler.gz')

if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)

lower_bounds = [0, 0, 0]
upper_bounds = [220+1, 154+1, 108+1]
res = 120
morse_set_data = np.loadtxt(mfile_path, delimiter=',', dtype=np.float64)
encoder_path = os.path.join(model_dir, 'encoder.pt')
dynamics_path = os.path.join(model_dir, 'dynamics.pt')

periodic_pts = {
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
}


class LeslieModel3D_Vectorized:
    def __init__(self, th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.survival_p1 = survival_p1
        self.survival_p2 = survival_p2

    def iterate(self, X, iterations=1):
        curr_X = X.copy()
        for _ in range(iterations):
            x0 = curr_X[:, 0]
            x1 = curr_X[:, 1]
            x2 = curr_X[:, 2]
            sum_x = x0 + x1 + x2
            next_x0 = (self.th1*x0 + self.th2*x1 + self.th3*x2) * np.exp(-0.1 * sum_x)
            next_x1 = self.survival_p1 * x0
            next_x2 = self.survival_p2 * x1
            curr_X = np.stack([next_x0, next_x1, next_x2], axis=1)
        return curr_X


class Box:
    def __init__(self, ID, lower_x, lower_y, upper_x, upper_y, M_label):
        self.ID = ID
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y
        self.M_label = M_label


class Edge:
    def __init__(self, v1, v2):
        if v1 < v2:
            self.u = tuple(v1)
            self.v = tuple(v2)
        else:
            self.u = tuple(v2)
            self.v = tuple(v1)
        if abs(self.u[1] - self.v[1]) < 1e-9:
            self.orientation = 'horizontal'
        elif abs(self.u[0] - self.v[0]) < 1e-9:
            self.orientation = 'vertical'
        else:
            self.orientation = 'diagonal'

    def __eq__(self, other):
        return isinstance(other, Edge) and self.u == other.u and self.v == other.v

    def __hash__(self):
        return hash((self.u, self.v))


class MorseSet:
    def __init__(self, file_path, label):
        self.label = int(label)
        self.boxes = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        self._load_from_file(file_path)

    def _load_from_file(self, file_path):
        try:
            data = np.loadtxt(file_path, delimiter=',', ndmin=2)
            mask = np.isclose(data[:, 4], self.label)
            filtered_data = data[mask]
            for i, row in enumerate(filtered_data):
                self.boxes.append(Box(
                    ID=i,
                    lower_x=row[0],
                    lower_y=row[1],
                    upper_x=row[2],
                    upper_y=row[3],
                    M_label=int(row[4])
                ))
        except Exception as e:
            print(f"[Error] Failed to load Morse Set {self.label}: {e}")
            self.boxes = []

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)

    def get_morse_set_boundary(self):
        boundary_edges = set()
        for box in self.boxes:
            p1 = (box.lower_x, box.lower_y)
            p2 = (box.upper_x, box.lower_y)
            p3 = (box.upper_x, box.upper_y)
            p4 = (box.lower_x, box.upper_y)
            for edge in [Edge(p1, p2), Edge(p2, p3), Edge(p3, p4), Edge(p4, p1)]:
                if edge in boundary_edges:
                    boundary_edges.remove(edge)
                else:
                    boundary_edges.add(edge)
        return boundary_edges


def encode_grid(lower_bounds, upper_bounds, res, encoder_path):
    grid = np.mgrid[
        lower_bounds[0]:upper_bounds[0]:(upper_bounds[0]-lower_bounds[0])/res,
        lower_bounds[1]:upper_bounds[1]:(upper_bounds[1]-lower_bounds[1])/res,
        lower_bounds[2]:upper_bounds[2]:(upper_bounds[2]-lower_bounds[2])/res
    ]
    grid_points = grid.reshape(3, -1).T
    mid_idx = len(grid_points) // 2
    image_points = grid_points.copy()

    scaler = joblib.load(scaler_path)
    model = LeslieModel3D_Vectorized()
    image_points[:mid_idx] = model.iterate(grid_points[:mid_idx], iterations=20)

    scaled_image = scaler.transform(image_points)
    image_tensor = torch.from_numpy(scaled_image).float()

    encoder = torch.load(encoder_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        latent_image_np = encoder(image_tensor.to(device)).cpu().numpy()

    return image_points, latent_image_np


def get_spatial_filter_mask(latent_points_np, morse_set_data):
    global_lx = np.min(morse_set_data[:, 0])
    global_ly = np.min(morse_set_data[:, 1])
    global_ux = np.max(morse_set_data[:, 2])
    global_uy = np.max(morse_set_data[:, 3])
    print(f"Morse Domain: X=[{global_lx}, {global_ux}], Y=[{global_ly}, {global_uy}]")

    px = latent_points_np[:, 0]
    py = latent_points_np[:, 1]
    print(f"Latent domain: X=[{np.min(px)}, {np.max(px)}], Y=[{np.min(py)}, {np.max(py)}]")

    return (px >= global_lx) & (px <= global_ux) & (py >= global_ly) & (py <= global_uy)


def label_3D_pts_vectorized(latent_points_np, grid_points, morse_set_data):
    num_pts = latent_points_np.shape[0]
    assigned_labels = np.full(num_pts, -1, dtype=int)
    px = latent_points_np[:, 0]
    py = latent_points_np[:, 1]

    for datapt in tqdm(morse_set_data, desc="Processing Morse Boxes"):
        lx, ly, ux, uy, label = datapt
        mask = (px >= lx) & (px <= ux) & (py >= ly) & (py <= uy)
        assigned_labels[mask] = int(label)

    valid_mask = (assigned_labels >= 0)
    x_list = grid_points[valid_mask, 0]
    y_list = grid_points[valid_mask, 1]
    z_list = grid_points[valid_mask, 2]
    labels = assigned_labels[valid_mask]
    print(set(labels))
    return x_list, y_list, z_list, labels


def label_3D_pts_with_filter(latent_points_np, grid_points, morse_set_data):
    spatial_mask = get_spatial_filter_mask(latent_points_np, morse_set_data)
    filtered_latent = latent_points_np[spatial_mask]
    filtered_grid = grid_points[spatial_mask]
    print(f"Points remaining after spatial filtering: {len(filtered_latent)} / {len(latent_points_np)}", flush=True)
    return label_3D_pts_vectorized(filtered_latent, filtered_grid, morse_set_data)


def find_and_save_preimage_samples(target_k, num_samples=100, iterations=21, batch_size=10):
    """
    Randomly samples 3D points until num_samples points are found that land in Morse set k.
    Saves the collection of points as a single NumPy array in a pickle file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.eval()
    model = LeslieModel3D_Vectorized()

    found_points = []
    attempt = 0
    search_lower = [0, 0, 0]
    search_upper = [110, 80, 55]

    print(f"Searching for {num_samples} points in the preimage of Morse set {target_k}...")
    print('Using upper bounds: ', search_upper)

    while len(found_points) < num_samples:
        attempt += 1
        seeds = np.random.uniform(low=search_lower, high=search_upper, size=(batch_size, 3))
        iterated_seeds = model.iterate(seeds, iterations=iterations)

        scaled_iterated = scaler.transform(iterated_seeds)
        with torch.no_grad():
            latent_z = encoder(torch.from_numpy(scaled_iterated).float().to(device)).cpu().numpy()

        px, py = latent_z[:, 0], latent_z[:, 1]

        target_boxes = morse_set_data[morse_set_data[:, 4] == target_k]
        batch_matches = np.zeros(batch_size, dtype=bool)
        for box in target_boxes:
            lx, ly, ux, uy = box[0], box[1], box[2], box[3]
            batch_matches |= (px >= lx) & (px <= ux) & (py >= ly) & (py <= uy)

        if np.any(batch_matches):
            matches = iterated_seeds[batch_matches]
            found_points.extend(matches)
            print(f"  Found {len(matches)} matches in batch {attempt}. Total: {min(len(found_points), num_samples)}/{num_samples}")

        if attempt % 10 == 0 and len(found_points) == 0:
            print(f"  Still searching... Processed {attempt * batch_size} points without a match.")

    final_points = np.array(found_points[:num_samples])
    save_file = os.path.join(plot_data_dir, f'preimage_samples_k{target_k}_{num_samples}pts.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(final_points, f)

    print(f"Success! {num_samples} points saved to: {save_file}")
    return final_points


def get_max_dynamics_loss(target_label, x_list, y_list, z_list, labels):
    indices = np.where(labels == target_label)[0]

    if len(indices) == 0:
        print(f"No points found with label {target_label}")
        return 0.0

    x_arr, y_arr, z_arr = np.array(x_list), np.array(y_list), np.array(z_list)
    current_pts = np.column_stack((x_arr[indices], y_arr[indices], z_arr[indices]))

    model_phys = LeslieModel3D_Vectorized()
    true_next_pts = model_phys.iterate(current_pts, iterations=1)

    scaler = joblib.load(scaler_path)
    current_scaled = scaler.transform(current_pts)
    true_next_scaled = scaler.transform(true_next_pts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.eval()
    dynamics = torch.load(dynamics_path, map_location=device, weights_only=False)
    dynamics.eval()

    with torch.no_grad():
        z_current = encoder(torch.from_numpy(current_scaled).float().to(device))
        z_next_true_encoded = encoder(torch.from_numpy(true_next_scaled).float().to(device))
        z_next_pred = dynamics(z_current)
        diff = z_next_true_encoded - z_next_pred
        loss_per_point = torch.sum(diff ** 2, dim=1).cpu().numpy()

    max_loss = np.max(loss_per_point)
    mean_loss = np.mean(loss_per_point)
    print(f"--- Dynamics Loss (L3) Analysis for Morse set {target_label} ---")
    print(f"Points analyzed: {len(indices)}")
    print(f"Max Loss: {max_loss:.6f}")
    print(f"Mean Loss: {mean_loss:.6f}")
    return max_loss


def is_in_range(point, edge):
    px, py = point
    (ux, uy) = edge.u
    (vx, vy) = edge.v
    eps = 1e-9
    if edge.orientation == 'horizontal':
        return (min(ux, vx) - eps <= px <= max(ux, vx) + eps)
    elif edge.orientation == 'vertical':
        return (min(uy, vy) - eps <= py <= max(uy, vy) + eps)
    return False


def get_orthogonal_distance(point, edge):
    px, py = point
    (ux, uy) = edge.u
    if edge.orientation == 'horizontal':
        return abs(py - uy)
    elif edge.orientation == 'vertical':
        return abs(px - ux)
    return float('inf')


def distance_point_to_boundary(point, boundary_edges):
    min_dist = float('inf')
    for edge in boundary_edges:
        if is_in_range(point, edge):
            dist = get_orthogonal_distance(point, edge)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def compute_min_boundary_separation(morse_set, dynamics_model, device):
    boundary_edges = morse_set.get_morse_set_boundary()

    unique_vertices = set()
    for box in morse_set.boxes:
        unique_vertices.add((box.lower_x, box.lower_y))
        unique_vertices.add((box.upper_x, box.lower_y))
        unique_vertices.add((box.upper_x, box.upper_y))
        unique_vertices.add((box.lower_x, box.upper_y))

    if not unique_vertices:
        print("[Warning] Morse set has no vertices.")
        return 0.0

    vertices_arr = np.array(list(unique_vertices), dtype=np.float32)

    dynamics_model.eval()
    with torch.no_grad():
        mapped_vertices = dynamics_model(torch.from_numpy(vertices_arr).to(device)).cpu().numpy()

    global_min_distance = float('inf')
    valid_points_count = 0
    for v_mapped in mapped_vertices:
        d = distance_point_to_boundary(tuple(v_mapped), boundary_edges)
        if d < global_min_distance:
            global_min_distance = d
        if d != float('inf'):
            valid_points_count += 1

    print(f"Evaluated {len(vertices_arr)} vertices. {valid_points_count} fell within orthogonal range of boundary.")
    return global_min_distance


def plot_latent_trajectory_small_pts(morse_set_data, periodic_pts, encoder_path, dynamics_path,
                                     scaler_path, save_filename, color_list=None, trajectory_steps=4):
    """
    Plots colored Morse sets with trajectory points.
    If trajectory_steps < 5: uniform black styling. If >= 5: markers grow and fade gray to black.
    """
    if color_list is None:
        color_list = ['#ffb000', '#fe6100', '#dc267f', '#648fff', '#785ef0', '#008080']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    dynamics = torch.load(dynamics_path, map_location=device, weights_only=False)
    encoder.eval(); dynamics.eval()

    lx, ly, ux, uy = morse_set_data[:, 0], morse_set_data[:, 1], morse_set_data[:, 2], morse_set_data[:, 3]
    labels = morse_set_data[:, 4].astype(int)
    centers_x, centers_y = (lx + ux) / 2, (ly + uy) / 2

    plt.figure(figsize=(10, 8))
    for lbl in np.unique(labels):
        mask = (labels == lbl)
        plt.scatter(centers_x[mask], centers_y[mask], color=color_list[lbl % len(color_list)],
                    marker='s', s=12, alpha=1, edgecolors='none', label=f'Morse set {lbl}', zorder=1)

    markers = ['s', '*', 'D', '*', '^', 'p']
    gray_to_black = mcolors.LinearSegmentedColormap.from_list("gb", ["#cccccc", "#000000"])
    is_short = trajectory_steps < 5

    with torch.no_grad():
        for label, points in periodic_pts.items():
            m_shape = markers[int(label) % len(markers)]
            pt_scaled = scaler.transform([points[0]])
            z = encoder(torch.from_numpy(pt_scaled).float().to(device))

            trajectory = [z.cpu().numpy()[0]]
            for _ in range(trajectory_steps):
                z = dynamics(z)
                trajectory.append(z.cpu().numpy()[0])

            traj_np = np.array(trajectory)
            plt.plot(traj_np[:, 0], traj_np[:, 1], color='black',
                     alpha=0.3 if is_short else 0.1, linestyle='-', linewidth=0.8, zorder=5)

            for i in range(len(traj_np)):
                prog = i / (len(traj_np) - 1) if len(traj_np) > 1 else 1.0
                if is_short:
                    size, current_color, lw, arrow_alpha = 20, 'black', 1.0, 1
                else:
                    size = 25 + (prog * 45)
                    current_color = gray_to_black(prog)
                    lw = 0.5 + (prog * 0.7)
                    arrow_alpha = 0.2 + (prog * 0.4)

                plt.scatter(traj_np[i, 0], traj_np[i, 1], facecolor=current_color,
                            marker=m_shape, s=size, edgecolors='black', linewidths=lw, zorder=10 + i)

                if i < len(traj_np) - 1:
                    plt.annotate('', xy=(traj_np[i+1, 0], traj_np[i+1, 1]),
                                 xytext=(traj_np[i, 0], traj_np[i, 1]),
                                 arrowprops=dict(arrowstyle='-|>', color='black',
                                                 lw=0.8, alpha=arrow_alpha, mutation_scale=10),
                                 zorder=100)

    plt.xlabel('$z_1$', fontsize=16); plt.ylabel('$z_2$', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    plt.show()


indexed_data_path = os.path.join(plot_data_dir, 'preimage_plot_data_indexed.pkl')

if not os.path.exists(indexed_data_path):
    image_points, latent_points_np = encode_grid(lower_bounds, upper_bounds, res, encoder_path)
    x_list, y_list, z_list, labels = label_3D_pts_with_filter(latent_points_np, image_points, morse_set_data)
    with open(indexed_data_path, 'wb') as f:
        pickle.dump({'x': x_list, 'y': y_list, 'z': z_list, 'labels': labels}, f)

with open(indexed_data_path, 'rb') as f:
    data = pickle.load(f)
    x_list, y_list, z_list, labels = data['x'], data['y'], data['z'], data['labels']

preimage_k4_path = os.path.join(plot_data_dir, 'preimage_samples_k4_20pts.pkl')
if not os.path.exists(preimage_k4_path):
    find_and_save_preimage_samples(target_k=4, num_samples=20, iterations=20, batch_size=100)

with open(preimage_k4_path, 'rb') as f:
    k4_samples = pickle.load(f)

x_list = np.concatenate([x_list, k4_samples[:, 0]])
y_list = np.concatenate([y_list, k4_samples[:, 1]])
z_list = np.concatenate([z_list, k4_samples[:, 2]])
labels = np.concatenate([labels, np.full(len(k4_samples), 4, dtype=int)])

max_semi_conj_error = get_max_dynamics_loss(m_label, x_list, y_list, z_list, labels)

M = MorseSet(mfile_path, m_label)
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

plot_latent_trajectory_small_pts(
    morse_set_data, periodic_pts, encoder_path, dynamics_path, scaler_path,
    os.path.join(save_path, "latent_trajectory.png"), color_list=color_list, trajectory_steps=4
)
