import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import g_tr as gt

# --- UPGRADED: Increased point density for better accuracy --- #
N = 5000
Nb = 2000
dataname = 'vk_plate_20000pts'
# ----------------------------------------------------------- #

# Generate interior collocation points in [0,1]^2
domain_data = uniform.rvs(size=(N, 2))
print(f"Interior domain data shape: {domain_data.shape}")

def generate_random_bdry(Nb):

    bdry_col = uniform.rvs(size=(Nb, 2))
    edge_choice = np.random.randint(0, 4, size=Nb)
    
    # Assign points to one of the four edges
    # 0: left (x=0), 1: right (x=1), 2: bottom (y=0), 3: top (y=1)
    bdry_col[edge_choice == 0, 0] = 0.0
    bdry_col[edge_choice == 1, 0] = 1.0
    bdry_col[edge_choice == 2, 1] = 0.0
    bdry_col[edge_choice == 3, 1] = 1.0
    
    return bdry_col

def compute_normals(bdry_col, eps=1e-8):

    x, y = bdry_col[:, 0], bdry_col[:, 1]
    n1, n2 = np.zeros_like(x), np.zeros_like(y)
    n1[np.isclose(x, 0.0)] = -1.0
    n1[np.isclose(x, 1.0)] = 1.0
    n2[np.isclose(y, 0.0)] = -1.0
    n2[np.isclose(y, 1.0)] = 1.0
    return n1.reshape(-1,1), n2.reshape(-1,1)

bdry_col = generate_random_bdry(Nb)
n1_np, n2_np = compute_normals(bdry_col)
normal_vec = np.hstack([n1_np, n2_np])
print(f"Boundary data shape: {bdry_col.shape}")
print(f"Normal vector shape: {normal_vec.shape}")

os.makedirs('dataset/', exist_ok=True)
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)
    pkl.dump(normal_vec,pfile)

u_gt, v_gt, f_gt, source_v_gt = gt.data_gen_interior(domain_data)
g1_data, h1_data, g2_data, h2_data = gt.data_gen_bdry(bdry_col, normal_vec)

with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    for item in [u_gt, v_gt, f_gt, source_v_gt, g1_data, h1_data, g2_data, h2_data]:
        pkl.dump(item, pfile)

print("Data generation complete and saved.")