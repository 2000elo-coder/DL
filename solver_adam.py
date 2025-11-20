import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import model, pde, tools, g_tr
import os
from time import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float32)

# --- CONFIGURATION ---
dataname = 'vk_plate_20000pts'
results_dir = 'results_vk_upgraded/'
os.makedirs(results_dir, exist_ok=True)
log_file_path = os.path.join(results_dir, "result.txt")

# Clear previous log file
with open(log_file_path, 'w') as f:
    f.write("Starting Training...\n")


def log_print(text):
    """Helper to print to console and save to text file"""
    print(text)
    with open(log_file_path, 'a') as f:
        f.write(text + "\n")


def format_time(seconds):
    h = int(seconds // 3600);
    m = int((seconds % 3600) // 60);
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- 1. Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn_model = model.VonKarmanPINN().to(device)
pinn_model.apply(model.init_weights)
log_print(f"PINN Model initialized on device: {device}")

# --- 2. Hyperparameters ---
lambda_pde = 1.0
lambda_bc_final_lbfgs = 10.0
learning_rate_adam = 5e-5
# UPGRADED EPOCHS
adam_epochs = 25000
lbfgs_total_iters = 15000
lbfgs_max_iter_per_step = 500
lbfgs_outer_steps = int(lbfgs_total_iters / lbfgs_max_iter_per_step)

batch_size = 1024
checkpoint_path = os.path.join(results_dir, "latest_checkpoint.pth")
save_every_adam = 1000

# --- 3. Load Data ---
with open("dataset/" + dataname, 'rb') as pfile:
    int_col, bdry_col, normal_vec = pkl.load(pfile), pkl.load(pfile), pkl.load(pfile)
log_print(f"Loaded collocation points.")

intx1, intx2 = np.split(int_col, 2, axis=1);
bdx1, bdx2 = np.split(bdry_col, 2, axis=1);
nx1, nx2 = np.split(normal_vec, 2, axis=1)
tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2 = tools.from_numpy_to_tensor(
    [intx1, intx2, bdx1, bdx2, nx1, nx2], [True, True, True, True, False, False], dtype=torch.float32)

with open("dataset/gt_on_{}".format(dataname), 'rb') as pfile:
    data_list = [pkl.load(pfile) for _ in range(8)]
u_gt_t, v_gt_t, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t = tools.from_numpy_to_tensor(
    data_list, [False] * 8, dtype=torch.float32)

# Move all tensors to device
tensors_to_move = [tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t]
tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t = [t.to(device) for t in
                                                                                     tensors_to_move]
log_print("Loaded all data and moved to device.")

# --- 4. Optimizer and State Initialization ---
optimizer_adam = opt.Adam(pinn_model.parameters(), lr=learning_rate_adam)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=1000, factor=0.5, min_lr=1e-7)
optimizer_lbfgs = torch.optim.LBFGS(pinn_model.parameters(), lr=0.1, max_iter=lbfgs_max_iter_per_step, history_size=100,
                                    line_search_fn="strong_wolfe")
interior_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tintx1, tintx2, f_t, source_v_t),
                                              batch_size=batch_size, shuffle=True)

start_epoch = 1;
start_lbfgs_step = 1;
skip_adam = False
loss_list, pde_loss_list, bc_loss_list = [], [], []


# --- HELPER: Exact Solution & Error Calculation ---
def get_exact_derivatives(x1, x2):
    """
    Calculates u_gt, v_gt and their 1st/2nd derivatives analytically
    for Error Norm calculation.
    u = sin^2(pi*x)sin^2(pi*y)
    v = (x^2-1)^2(y^2-1)^2
    """
    pi = np.pi
    # u derivatives
    sin_pix = torch.sin(pi * x1);
    cos_pix = torch.cos(pi * x1)
    sin_piy = torch.sin(pi * x2);
    cos_piy = torch.cos(pi * x2)

    u_exact = (sin_pix ** 2) * (sin_piy ** 2)
    u_x = 2 * pi * sin_pix * cos_pix * (sin_piy ** 2)
    u_y = (sin_pix ** 2) * 2 * pi * sin_piy * cos_piy
    u_xx = 2 * pi ** 2 * (cos_pix ** 2 - sin_pix ** 2) * (sin_piy ** 2)
    u_yy = (sin_pix ** 2) * 2 * pi ** 2 * (cos_piy ** 2 - sin_piy ** 2)
    u_xy = 4 * pi ** 2 * sin_pix * cos_pix * sin_piy * cos_piy

    # v derivatives
    X = x1;
    Y = x2
    term_x = (X ** 2 - 1) ** 2;
    term_y = (Y ** 2 - 1) ** 2
    d_term_x = 2 * (X ** 2 - 1) * 2 * X;
    dd_term_x = 4 * (3 * X ** 2 - 1)
    d_term_y = 2 * (Y ** 2 - 1) * 2 * Y;
    dd_term_y = 4 * (3 * Y ** 2 - 1)

    v_exact = term_x * term_y
    v_x = d_term_x * term_y
    v_y = term_x * d_term_y
    v_xx = dd_term_x * term_y
    v_yy = term_x * dd_term_y
    v_xy = d_term_x * d_term_y

    return u_exact, u_x, u_y, u_xx, u_yy, u_xy, v_exact, v_x, v_y, v_xx, v_yy, v_xy


def compute_error_norms(model, x1_eval, x2_eval):
    """Computes L2, H1, H2 errors for u and v"""
    model.eval()

    # Ground Truth
    u_gt, u_x_gt, u_y_gt, u_xx_gt, u_yy_gt, u_xy_gt, \
        v_gt, v_x_gt, v_y_gt, v_xx_gt, v_yy_gt, v_xy_gt = get_exact_derivatives(x1_eval, x2_eval)

    # Predictions (require grad for derivatives)
    u_pred, v_pred = model(x1_eval, x2_eval)

    # 1st Derivatives
    u_x_pred = torch.autograd.grad(u_pred.sum(), x1_eval, create_graph=True)[0]
    u_y_pred = torch.autograd.grad(u_pred.sum(), x2_eval, create_graph=True)[0]
    v_x_pred = torch.autograd.grad(v_pred.sum(), x1_eval, create_graph=True)[0]
    v_y_pred = torch.autograd.grad(v_pred.sum(), x2_eval, create_graph=True)[0]

    # 2nd Derivatives
    u_xx_pred = torch.autograd.grad(u_x_pred.sum(), x1_eval, create_graph=True)[0]
    u_yy_pred = torch.autograd.grad(u_y_pred.sum(), x2_eval, create_graph=True)[0]
    u_xy_pred = torch.autograd.grad(u_x_pred.sum(), x2_eval, create_graph=True)[0]

    v_xx_pred = torch.autograd.grad(v_x_pred.sum(), x1_eval, create_graph=True)[0]
    v_yy_pred = torch.autograd.grad(v_y_pred.sum(), x2_eval, create_graph=True)[0]
    v_xy_pred = torch.autograd.grad(v_x_pred.sum(), x2_eval, create_graph=True)[0]

    # L2 Error
    def relative_l2(pred, true):
        return torch.sqrt(torch.mean((pred - true) ** 2)) / torch.sqrt(torch.mean(true ** 2))

    l2_u = relative_l2(u_pred, u_gt)
    l2_v = relative_l2(v_pred, v_gt)

    # H1 Error (Squared sums of diffs / Squared sums of true H1)
    # Approx calculation: ||u-u_h||_H1 approx sqrt(L2^2 + |Du - Du_h|^2)
    diff_u_H1 = (u_pred - u_gt) ** 2 + (u_x_pred - u_x_gt) ** 2 + (u_y_pred - u_y_gt) ** 2
    true_u_H1 = u_gt ** 2 + u_x_gt ** 2 + u_y_gt ** 2
    h1_u = torch.sqrt(torch.mean(diff_u_H1)) / torch.sqrt(torch.mean(true_u_H1))

    diff_v_H1 = (v_pred - v_gt) ** 2 + (v_x_pred - v_x_gt) ** 2 + (v_y_pred - v_y_gt) ** 2
    true_v_H1 = v_gt ** 2 + v_x_gt ** 2 + v_y_gt ** 2
    h1_v = torch.sqrt(torch.mean(diff_v_H1)) / torch.sqrt(torch.mean(true_v_H1))

    # H2 Error
    diff_u_H2 = diff_u_H1 + (u_xx_pred - u_xx_gt) ** 2 + (u_yy_pred - u_yy_gt) ** 2 + 2 * (u_xy_pred - u_xy_gt) ** 2
    true_u_H2 = true_u_H1 + u_xx_gt ** 2 + u_yy_gt ** 2 + 2 * u_xy_gt ** 2
    h2_u = torch.sqrt(torch.mean(diff_u_H2)) / torch.sqrt(torch.mean(true_u_H2))

    diff_v_H2 = diff_v_H1 + (v_xx_pred - v_xx_gt) ** 2 + (v_yy_pred - v_yy_gt) ** 2 + 2 * (v_xy_pred - v_xy_gt) ** 2
    true_v_H2 = true_v_H1 + v_xx_gt ** 2 + v_yy_gt ** 2 + 2 * v_xy_gt ** 2
    h2_v = torch.sqrt(torch.mean(diff_v_H2)) / torch.sqrt(torch.mean(true_v_H2))

    model.train()  # Switch back to train
    return l2_u.item(), h1_u.item(), h2_u.item(), l2_v.item(), h1_v.item(), h2_v.item()


# Load Checkpoint Logic
if os.path.exists(checkpoint_path):
    log_print(f"--- Checkpoint found. Loading state... ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer_type = checkpoint.get('optimizer_type', 'adam')
    pinn_model.load_state_dict(checkpoint['model_state_dict'])
    loss_list, pde_loss_list, bc_loss_list = [checkpoint.get(k, []) for k in
                                              ['loss_history', 'pde_loss_history', 'bc_loss_history']]
    if optimizer_type == 'lbfgs':
        log_print("Optimizer type is L-BFGS. Skipping Adam phase.")
        optimizer_lbfgs.load_state_dict(checkpoint['optimizer_state_dict'])
        start_lbfgs_step = checkpoint['lbfgs_step'] + 1
        skip_adam = True
    else:
        log_print(f"Resuming Adam training from epoch {checkpoint['epoch']}.")
        optimizer_adam.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
else:
    log_print("--- No checkpoint found. Starting training from scratch. ---")

# --- 5. Phase 1: Adam Training ---
training_start_time = time()
if not skip_adam:
    log_print(f"\n--- Phase 1: Training with Adam ({adam_epochs} epochs) ---")
    for epoch in range(start_epoch, adam_epochs + 1):
        current_lambda_bc = 10.0 + 90.0 * max(0, 1 - (epoch / adam_epochs))
        pinn_model.train()
        total_loss_epoch, pde_loss_epoch, bc_loss_epoch = 0.0, 0.0, 0.0

        for batch_x1, batch_x2, batch_f, batch_sv in interior_loader:
            optimizer_adam.zero_grad()
            batch_x1.requires_grad_();
            batch_x2.requires_grad_()
            loss, pde_loss, bc_loss = pde.total_loss(
                pinn_model.net_u, pinn_model.net_v, batch_x1, batch_x2, batch_f, batch_sv,
                tbdx1, tbdx2, tnx1, tnx2, g1_t, h1_t, g2_t, h2_t, lambda_pde, current_lambda_bc)
            loss.backward();
            optimizer_adam.step()
            total_loss_epoch += loss.item();
            pde_loss_epoch += pde_loss.item();
            bc_loss_epoch += bc_loss.item()

        num_batches = len(interior_loader)
        avg_loss, avg_pde, avg_bc = total_loss_epoch / num_batches, pde_loss_epoch / num_batches, bc_loss_epoch / num_batches
        loss_list.append(avg_loss);
        pde_loss_list.append(avg_pde);
        bc_loss_list.append(avg_bc)
        scheduler.step(avg_loss)

        if epoch % 100 == 0 or epoch == start_epoch:
            elapsed = time() - training_start_time
            etr = (elapsed / (epoch - start_epoch + 1)) * (adam_epochs - epoch)

            # Compute Errors
            l2u, h1u, h2u, l2v, h1v, h2v = compute_error_norms(pinn_model, tintx1, tintx2)

            log_str = (f"Adam {epoch:5d} | L: {avg_loss:.3e} | PDE: {avg_pde:.3e} | BC: {avg_bc:.3e} | "
                       f"U[L2:{l2u:.2e} H1:{h1u:.2e} H2:{h2u:.2e}] V[L2:{l2v:.2e}] | ETR: {format_time(etr)}")
            log_print(log_str)

        if epoch % save_every_adam == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': pinn_model.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'loss_history': loss_list,
                'pde_loss_history': pde_loss_list, 'bc_loss_history': bc_loss_list, 'optimizer_type': 'adam'
            }, checkpoint_path)

# --- 6. Phase 2: L-BFGS Training ---
log_print(f"\n--- Phase 2: Fine-Tuning with L-BFGS ({lbfgs_outer_steps} steps x {lbfgs_max_iter_per_step} iters) ---")
global_step = len(loss_list)


def closure_lbfgs():
    optimizer_lbfgs.zero_grad()
    loss, pde_loss, bc_loss = pde.total_loss(
        pinn_model.net_u, pinn_model.net_v, tintx1, tintx2, f_t, source_v_t,
        tbdx1, tbdx2, tnx1, tnx2, g1_t, h1_t, g2_t, h2_t, lambda_pde, lambda_bc_final_lbfgs)
    loss.backward()
    loss_list.append(loss.item());
    pde_loss_list.append(pde_loss.item());
    bc_loss_list.append(bc_loss.item())
    return loss


for step in range(start_lbfgs_step, lbfgs_outer_steps + 1):
    pinn_model.train()
    optimizer_lbfgs.step(closure_lbfgs)

    # Evaluation after step
    curr_loss = loss_list[-1]
    l2u, h1u, h2u, l2v, h1v, h2v = compute_error_norms(pinn_model, tintx1, tintx2)
    log_print(f"L-BFGS Step {step}/{lbfgs_outer_steps} | Loss: {curr_loss:.3e} | "
              f"U[L2:{l2u:.2e} H1:{h1u:.2e} H2:{h2u:.2e}] V[L2:{l2v:.2e}]")

    torch.save({
        'lbfgs_step': step, 'model_state_dict': pinn_model.state_dict(),
        'optimizer_state_dict': optimizer_lbfgs.state_dict(),
        'loss_history': loss_list, 'pde_loss_history': pde_loss_list,
        'bc_loss_history': bc_loss_list, 'optimizer_type': 'lbfgs'
    }, checkpoint_path)

log_print(f"\nTraining finished in {format_time(time() - training_start_time)}.")

# --- 7. Final Evaluation and Detailed Plotting ---
log_print("Generating final high-res plots...")
resolution = 200
x_pts, y_pts = np.linspace(0, 1, resolution), np.linspace(0, 1, resolution)
ms_x, ms_y = np.meshgrid(x_pts, y_pts)
plot_xy_flat = np.hstack((ms_x.flatten()[:, None], ms_y.flatten()[:, None]))

u_gt_plot, v_gt_plot, _, _ = g_tr.data_gen_interior(plot_xy_flat)
ms_u_gt = u_gt_plot.reshape(ms_x.shape)
ms_v_gt = v_gt_plot.reshape(ms_x.shape)

t_plot_x = Variable(torch.from_numpy(plot_xy_flat[:, 0:1]).float()).to(device)
t_plot_y = Variable(torch.from_numpy(plot_xy_flat[:, 1:2]).float()).to(device)
pinn_model.eval()
with torch.no_grad():
    u_pred_plot = pinn_model.net_u(t_plot_x, t_plot_y).cpu().numpy()
    v_pred_plot = pinn_model.net_v(t_plot_x, t_plot_y).cpu().numpy()
ms_u_pred = u_pred_plot.reshape(ms_x.shape)
ms_v_pred = v_pred_plot.reshape(ms_x.shape)

# Calculate Errors for Plotting
error_u = np.abs(ms_u_gt - ms_u_pred)
error_v = np.abs(ms_v_gt - ms_v_pred)


def create_comprehensive_plot(variable_name, x, y, actual, pred, error):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Analysis of {variable_name}(x,y)', fontsize=20)

    # Row 1: 2D Plots
    ax1 = fig.add_subplot(231)
    im1 = ax1.contourf(x, y, actual, levels=50, cmap='jet')
    ax1.set_title(f'Actual {variable_name} (2D)');
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(232)
    im2 = ax2.contourf(x, y, pred, levels=50, cmap='jet')
    ax2.set_title(f'Predicted {variable_name} (2D)');
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(233)
    im3 = ax3.contourf(x, y, error, levels=50, cmap='inferno')
    ax3.set_title(f'Error |{variable_name} - {variable_name}_pred| (2D)');
    plt.colorbar(im3, ax=ax3)

    # Row 2: 3D Plots
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.plot_surface(x, y, actual, cmap='jet', edgecolor='none')
    ax4.set_title(f'Actual {variable_name} (3D)')

    ax5 = fig.add_subplot(235, projection='3d')
    ax5.plot_surface(x, y, pred, cmap='jet', edgecolor='none')
    ax5.set_title(f'Predicted {variable_name} (3D)')

    ax6 = fig.add_subplot(236, projection='3d')
    ax6.plot_surface(x, y, error, cmap='inferno', edgecolor='none')
    ax6.set_title(f'Error {variable_name} (3D)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, f'Analysis_{variable_name}.png'))
    plt.close()
    log_print(f"Saved plot for {variable_name}.")


create_comprehensive_plot("u", ms_x, ms_y, ms_u_gt, ms_u_pred, error_u)
create_comprehensive_plot("v", ms_x, ms_y, ms_v_gt, ms_v_pred, error_v)

log_print("All tasks completed.")