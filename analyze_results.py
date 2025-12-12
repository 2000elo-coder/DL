import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import sys
from torch.autograd import Variable

# Import your local modules
import model
import tools

# --- CONFIGURATION ---
RESULTS_DIR = 'results_vk_upgraded'
DATASET_PATH = 'dataset/vk_plate_20000pts'  # Must match the filename in data.py
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "latest_checkpoint.pth")
REPORT_PATH = os.path.join(RESULTS_DIR, "final_analysis_report.txt")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "loss_history_final.png")

# Set device and precision (Must match training precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def get_exact_derivatives(x1, x2):
    """
    Calculates analytical derivatives for u and v based on g_tr.py logic.
    u = (1 / 2pi^2) * sin^2(pi*x)sin^2(pi*y)
    v = (x^2-1)^2(y^2-1)^2
    """
    pi = np.pi
    u_factor = 1.0 / (2.0 * pi**2)

    # --- U Calculations ---
    sin_pix = torch.sin(pi * x1); cos_pix = torch.cos(pi * x1)
    sin_piy = torch.sin(pi * x2); cos_piy = torch.cos(pi * x2)

    # Raw u (sin^2 * sin^2)
    raw_u = (sin_pix ** 2) * (sin_piy ** 2)
    raw_u_x = 2 * pi * sin_pix * cos_pix * (sin_piy ** 2)
    raw_u_y = (sin_pix ** 2) * 2 * pi * sin_piy * cos_piy
    raw_u_xx = 2 * pi ** 2 * (cos_pix ** 2 - sin_pix ** 2) * (sin_piy ** 2)
    raw_u_yy = (sin_pix ** 2) * 2 * pi ** 2 * (cos_piy ** 2 - sin_piy ** 2)
    raw_u_xy = 4 * pi ** 2 * sin_pix * cos_pix * sin_piy * cos_piy

    # Apply Scaling Factor
    u_gt = u_factor * raw_u
    u_x = u_factor * raw_u_x
    u_y = u_factor * raw_u_y
    u_xx = u_factor * raw_u_xx
    u_yy = u_factor * raw_u_yy
    u_xy = u_factor * raw_u_xy

    # --- V Calculations ---
    X = x1; Y = x2
    term_x = (X ** 2 - 1) ** 2
    term_y = (Y ** 2 - 1) ** 2
    d_term_x = 2 * (X ** 2 - 1) * 2 * X
    dd_term_x = 4 * (3 * X ** 2 - 1)
    d_term_y = 2 * (Y ** 2 - 1) * 2 * Y
    dd_term_y = 4 * (3 * Y ** 2 - 1)

    v_gt = term_x * term_y
    v_x = d_term_x * term_y
    v_y = term_x * d_term_y
    v_xx = dd_term_x * term_y
    v_yy = term_x * dd_term_y
    v_xy = d_term_x * d_term_y

    return u_gt, u_x, u_y, u_xx, u_yy, u_xy, v_gt, v_x, v_y, v_xx, v_yy, v_xy

def calculate_errors(net_model, x1, x2):
    """Compute L2, H1, H2 errors using Autograd"""
    net_model.eval()
    
    # 1. Get Ground Truth
    u_gt, u_x_gt, u_y_gt, u_xx_gt, u_yy_gt, u_xy_gt, \
    v_gt, v_x_gt, v_y_gt, v_xx_gt, v_yy_gt, v_xy_gt = get_exact_derivatives(x1, x2)

    # 2. Get Predictions and Derivatives
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    u_pred, v_pred = net_model(x1, x2)

    # Helper for gradient
    def get_grad(y, x):
        return torch.autograd.grad(y.sum(), x, create_graph=True)[0]

    # 1st Order
    u_x = get_grad(u_pred, x1); u_y = get_grad(u_pred, x2)
    v_x = get_grad(v_pred, x1); v_y = get_grad(v_pred, x2)

    # 2nd Order
    u_xx = get_grad(u_x, x1); u_yy = get_grad(u_y, x2); u_xy = get_grad(u_x, x2)
    v_xx = get_grad(v_x, x1); v_yy = get_grad(v_y, x2); v_xy = get_grad(v_x, x2)

    # 3. Compute Norms
    def sum_sq(t): return torch.sum(t ** 2)

    # U Errors
    diff_u_l2 = sum_sq(u_pred - u_gt)
    diff_u_h1 = diff_u_l2 + sum_sq(u_x - u_x_gt) + sum_sq(u_y - u_y_gt)
    diff_u_h2 = diff_u_h1 + sum_sq(u_xx - u_xx_gt) + sum_sq(u_yy - u_yy_gt) + 2*sum_sq(u_xy - u_xy_gt)
    
    norm_u_l2 = sum_sq(u_gt)
    norm_u_h1 = norm_u_l2 + sum_sq(u_x_gt) + sum_sq(u_y_gt)
    norm_u_h2 = norm_u_h1 + sum_sq(u_xx_gt) + sum_sq(u_yy_gt) + 2*sum_sq(u_xy_gt)

    # V Errors
    diff_v_l2 = sum_sq(v_pred - v_gt)
    diff_v_h1 = diff_v_l2 + sum_sq(v_x - v_x_gt) + sum_sq(v_y - v_y_gt)
    diff_v_h2 = diff_v_h1 + sum_sq(v_xx - v_xx_gt) + sum_sq(v_yy - v_yy_gt) + 2*sum_sq(v_xy - v_xy_gt)

    norm_v_l2 = sum_sq(v_gt)
    norm_v_h1 = norm_v_l2 + sum_sq(v_x_gt) + sum_sq(v_y_gt)
    norm_v_h2 = norm_v_h1 + sum_sq(v_xx_gt) + sum_sq(v_yy_gt) + 2*sum_sq(v_xy_gt)

    return {
        'u_L2': torch.sqrt(diff_u_l2 / norm_u_l2).item(),
        'u_H1': torch.sqrt(diff_u_h1 / norm_u_h1).item(),
        'u_H2': torch.sqrt(diff_u_h2 / norm_u_h2).item(),
        'v_L2': torch.sqrt(diff_v_l2 / norm_v_l2).item(),
        'v_H1': torch.sqrt(diff_v_h1 / norm_v_h1).item(),
        'v_H2': torch.sqrt(diff_v_h2 / norm_v_h2).item()
    }

def analyze_model_structure(net_model):
    """Counts parameters and creates architecture string"""
    total_params = 0
    nonzero_params = 0
    tolerance = 1e-8

    for param in net_model.parameters():
        flat = param.view(-1)
        total_params += flat.numel()
        nonzero_params += torch.sum(torch.abs(flat) > tolerance).item()

    arch_str = str(net_model)
    return arch_str, total_params, nonzero_params

def plot_loss_history(history_dict):
    """Plots and saves loss history"""
    loss = history_dict.get('loss_history', [])
    pde = history_dict.get('pde_loss_history', [])
    bc = history_dict.get('bc_loss_history', [])

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss, label='Total Loss', color='black', linewidth=1.5)
    if pde: plt.semilogy(pde, label='PDE Loss', linestyle='--', color='blue', alpha=0.7)
    if bc: plt.semilogy(bc, label='BC Loss', linestyle='--', color='red', alpha=0.7)
    
    plt.title('Training Loss History (Log Scale)')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    print(f"Loss plot saved to {LOSS_PLOT_PATH}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Post-Training Analysis ---")

    # 1. Initialize Model
    pinn_model = model.VonKarmanPINN().to(device)

    # 2. Load Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Handle loading (State Dict vs Full Model)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pinn_model.load_state_dict(checkpoint['model_state_dict'])
        loss_data = {
            'loss_history': checkpoint.get('loss_history', []),
            'pde_loss_history': checkpoint.get('pde_loss_history', []),
            'bc_loss_history': checkpoint.get('bc_loss_history', [])
        }
    else:
        print("Checkpoint format not recognized (expected dict with model_state_dict).")
        sys.exit(1)

    # 3. TASK 1: Plot Loss
    print("Generating loss plot...")
    plot_loss_history(loss_data)

    # 4. TASK 3: Model Architecture Analysis
    print("Analyzing model architecture...")
    arch_str, total_p, nonzero_p = analyze_model_structure(pinn_model)

    # 5. TASK 2: Error Calculation
    print("Loading test data for error calculation...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Data file {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH, 'rb') as f:
        # Load only the first item (interior points)
        int_col = pkl.load(f)

    # Prepare tensors
    intx1, intx2 = np.split(int_col, 2, axis=1)
    t_x1 = torch.tensor(intx1, dtype=torch.float32, device=device)
    t_x2 = torch.tensor(intx2, dtype=torch.float32, device=device)

    print("Computing L2, H1, H2 errors (this may take a moment)...")
    errors = calculate_errors(pinn_model, t_x1, t_x2)

    # 6. Generate Report
    print(f"Writing report to {REPORT_PATH}...")
    
    report_content = (
        "======================================================\n"
        "           VON KARMAN PINN - FINAL ANALYSIS           \n"
        "======================================================\n\n"
        "--- MODEL ARCHITECTURE ---\n"
        f"{arch_str}\n\n"
        f"Total Parameters:      {total_p}\n"
        f"Non-zero Parameters:   {nonzero_p}\n"
        f"Sparsity:              {100 * (1 - nonzero_p/total_p):.2f}%\n\n"
        "--- ERROR ANALYSIS ---\n"
        "Errors computed on interior domain points.\n\n"
        "DISPLACEMENT u(x,y):\n"
        f"  Relative L2 Error:   {errors['u_L2']:.6e}\n"
        f"  Relative H1 Error:   {errors['u_H1']:.6e}\n"
        f"  Relative H2 Error:   {errors['u_H2']:.6e}\n\n"
        "AIRY STRESS v(x,y):\n"
        f"  Relative L2 Error:   {errors['v_L2']:.6e}\n"
        f"  Relative H1 Error:   {errors['v_H1']:.6e}\n"
        f"  Relative H2 Error:   {errors['v_H2']:.6e}\n\n"
        "======================================================\n"
    )

    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)

    print("--- Analysis Complete ---")
    print(f"1. Loss Plot: {LOSS_PLOT_PATH}")
    print(f"2. Report:    {REPORT_PATH}")