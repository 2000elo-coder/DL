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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_dtype(torch.float32)

def format_time(seconds):
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- 1. Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn_model = model.VonKarmanPINN().to(device)
pinn_model.apply(model.init_weights)
print(f"PINN Model initialized on device: {device}")

dataname = 'vk_plate_20000pts'
results_dir = 'results_vk_upgraded/'
os.makedirs(results_dir, exist_ok=True)

# --- 2. Hyperparameters ---
lambda_pde = 1.0
lambda_bc_final_lbfgs = 10.0
learning_rate_adam = 5e-5
adam_epochs = 18100
batch_size = 1024
checkpoint_path = os.path.join(results_dir, "latest_checkpoint.pth")
save_every_adam = 1000
lbfgs_outer_steps = 35
lbfgs_max_iter_per_step = 500

# --- 3. Load Data ---
with open("dataset/"+dataname,'rb') as pfile:
    int_col, bdry_col, normal_vec = pkl.load(pfile), pkl.load(pfile), pkl.load(pfile)
print(f"Loaded collocation points.")

intx1,intx2 = np.split(int_col,2,axis=1); bdx1,bdx2 = np.split(bdry_col,2,axis=1); nx1,nx2 = np.split(normal_vec,2,axis=1)
tintx1,tintx2,tbdx1,tbdx2,tnx1,tnx2 = tools.from_numpy_to_tensor(
    [intx1,intx2,bdx1,bdx2,nx1,nx2], [True,True,True,True,False,False], dtype=torch.float32)

with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    data_list = [pkl.load(pfile) for _ in range(8)]
u_gt_t, v_gt_t, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t = tools.from_numpy_to_tensor(
    data_list, [False]*8, dtype=torch.float32)

# Move all tensors to the selected device
tensors_to_move = [tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t]
tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f_t, source_v_t, g1_t, h1_t, g2_t, h2_t = [t.to(device) for t in tensors_to_move]
print("Loaded all data and moved to device.")

# --- 4. Optimizer and State Initialization ---
optimizer_adam = opt.Adam(pinn_model.parameters(), lr=learning_rate_adam)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer_adam, patience=500, factor=0.5, min_lr=1e-7)
optimizer_lbfgs = torch.optim.LBFGS(pinn_model.parameters(), lr=0.1, max_iter=lbfgs_max_iter_per_step, max_eval=None, history_size=100, line_search_fn="strong_wolfe")
interior_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tintx1, tintx2, f_t, source_v_t), batch_size=batch_size, shuffle=True)
start_epoch = 1; start_lbfgs_step = 1; skip_adam = False
loss_list, pde_loss_list, bc_loss_list = [], [], []

if os.path.exists(checkpoint_path):
    print(f"--- Checkpoint found. Loading state... ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer_type = checkpoint.get('optimizer_type', 'adam')
    pinn_model.load_state_dict(checkpoint['model_state_dict'])
    loss_list, pde_loss_list, bc_loss_list = [checkpoint.get(k, []) for k in ['loss_history', 'pde_loss_history', 'bc_loss_history']]
    if optimizer_type == 'lbfgs':
        print("Optimizer type is L-BFGS. Skipping Adam phase.")
        optimizer_lbfgs.load_state_dict(checkpoint['optimizer_state_dict'])
        start_lbfgs_step = checkpoint['lbfgs_step'] + 1
        skip_adam = True
    else: # Adam
        print("Resuming Adam training.")
        optimizer_adam.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
else:
    print("--- No checkpoint found. Starting training from scratch. ---")

# --- 5. Phase 1: Adam Training ---
training_start_time = time()
if not skip_adam:
    print("\n--- Phase 1: Training with Adam Optimizer (with weight annealing) ---")
    for epoch in range(start_epoch, adam_epochs + 1):
        current_lambda_bc = 10.0 + 90.0 * max(0, 1 - (epoch / adam_epochs))
        pinn_model.train()
        total_loss_epoch, pde_loss_epoch, bc_loss_epoch = 0.0, 0.0, 0.0
        for batch_x1, batch_x2, batch_f, batch_sv in interior_loader:
            optimizer_adam.zero_grad()
            batch_x1.requires_grad_(); batch_x2.requires_grad_()
            loss, pde_loss, bc_loss = pde.total_loss(
                pinn_model.net_u, pinn_model.net_v, batch_x1, batch_x2, batch_f, batch_sv,
                tbdx1, tbdx2, tnx1, tnx2, g1_t, h1_t, g2_t, h2_t, lambda_pde, current_lambda_bc)
            loss.backward(); optimizer_adam.step()
            total_loss_epoch += loss.item(); pde_loss_epoch += pde_loss.item(); bc_loss_epoch += bc_loss.item()

        num_batches = len(interior_loader)
        avg_total_loss, avg_pde_loss, avg_bc_loss = total_loss_epoch/num_batches, pde_loss_epoch/num_batches, bc_loss_epoch/num_batches
        loss_list.append(avg_total_loss); pde_loss_list.append(avg_pde_loss); bc_loss_list.append(avg_bc_loss)
        scheduler.step(avg_total_loss)

        if epoch % 100 == 0 or epoch == start_epoch:
            elapsed = time() - training_start_time
            etr = (elapsed / (epoch - start_epoch + 1)) * (adam_epochs - epoch) if (epoch - start_epoch + 1) > 0 else 0
            print(f"Adam Epoch {epoch:5d}/{adam_epochs} | Total L: {avg_total_loss:.4e} | PDE L: {avg_pde_loss:.4e} | BC L: {avg_bc_loss:.4e} | λ_bc: {current_lambda_bc:.1f} | ETR: {format_time(etr)}")

        if epoch % save_every_adam == 0 or epoch == adam_epochs:
            torch.save({
                'epoch': epoch, 'model_state_dict': pinn_model.state_dict(),
                'optimizer_state_dict': optimizer_adam.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'loss_history': loss_list,
                'pde_loss_history': pde_loss_list, 'bc_loss_history': bc_loss_list, 'optimizer_type': 'adam'
            }, checkpoint_path)

# --- 6. Phase 2: L-BFGS Training ---
print("\n--- Phase 2: Fine-Tuning with L-BFGS Optimizer ---")
global_step = len(loss_list)
def closure_lbfgs():
    optimizer_lbfgs.zero_grad()
    loss, pde_loss, bc_loss = pde.total_loss(
        pinn_model.net_u, pinn_model.net_v, tintx1, tintx2, f_t, source_v_t,
        tbdx1, tbdx2, tnx1, tnx2, g1_t, h1_t, g2_t, h2_t, lambda_pde, lambda_bc_final_lbfgs)
    loss.backward()
    loss_list.append(loss.item()); pde_loss_list.append(pde_loss.item()); bc_loss_list.append(bc_loss.item())
    global global_step; global_step += 1
    if global_step % 100 == 0:
        print(f"  L-BFGS Eval {global_step:6d} | Total L: {loss.item():.4e} | PDE L: {pde_loss.item():.4e} | BC L: {bc_loss.item():.4e}")
    return loss

for step in range(start_lbfgs_step, lbfgs_outer_steps + 1):
    print(f"--- L-BFGS Outer Step: {step}/{lbfgs_outer_steps} ---")
    pinn_model.train(); optimizer_lbfgs.step(closure_lbfgs)
    torch.save({
        'lbfgs_step': step, 'model_state_dict': pinn_model.state_dict(),
        'optimizer_state_dict': optimizer_lbfgs.state_dict(),
        'loss_history': loss_list, 'pde_loss_history': pde_loss_list,
        'bc_loss_history': bc_loss_list, 'optimizer_type': 'lbfgs'
    }, checkpoint_path)
    print(f"--- L-BFGS checkpoint saved after step {step}. ---")

print(f"\nTraining finished in {format_time(time() - training_start_time)}.")

# --- 7. Final Evaluation and Plotting ---
print("Generating final plots and calculating L2 error...")
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

# L2 Error Calculation
l2_error_u = np.sqrt(np.mean((ms_u_gt - ms_u_pred)**2))
l2_relative_error_u = l2_error_u / (np.sqrt(np.mean(ms_u_gt**2)) + 1e-8)
l2_error_v = np.sqrt(np.mean((ms_v_gt - ms_v_pred)**2))
l2_relative_error_v = l2_error_v / (np.sqrt(np.mean(ms_v_gt**2)) + 1e-8)
print("\n--- Final L2 Errors ---")
print(f"L2 Relative Error (u): {l2_relative_error_u:.4e}")
print(f"L2 Relative Error (v): {l2_relative_error_v:.4e}\n")


# Combined Loss History Plot
plt.figure(figsize=(12, 7))
epochs_axis_total = range(1, len(loss_list) + 1)
plt.plot(epochs_axis_total, loss_list, label='Total Weighted Loss (L)', color='black', linewidth=2)
if len(pde_loss_list) < len(loss_list):
    start_idx = len(loss_list) - len(pde_loss_list)
    epochs_axis_components = epochs_axis_total[start_idx:]
    plt.plot(epochs_axis_components, pde_loss_list, label='PDE Loss (L_pde)', color='blue', linestyle='--')
    plt.plot(epochs_axis_components, bc_loss_list, label='BC Loss (L_bc)', color='green', linestyle='--')
else:
    plt.plot(epochs_axis_total, pde_loss_list, label='PDE Loss (L_pde)', color='blue', linestyle='--')
    plt.plot(epochs_axis_total, bc_loss_list, label='BC Loss (L_bc)', color='green', linestyle='--')
plt.yscale('log'); plt.title('Training Loss History', fontsize=16)
plt.xlabel('Optimization Step', fontsize=12); plt.ylabel('Loss Value (Log Scale)', fontsize=12)
plt.grid(True, which="both", ls=":"); plt.axvline(x=adam_epochs, color='r', linestyle='-.', label='Switch to L-BFGS')
plt.legend(fontsize=12); plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'loss_components_history.png'))
plt.close()
print(f"Loss history plot saved to {os.path.join(results_dir, 'loss_components_history.png')}")

# 3D Surface Plot for v(x,y)
fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ms_x, ms_y, ms_v_pred, cmap='jet', edgecolor='none')
ax.set_title('PINN Solution for Airy Stress Function v(x,y)', fontsize=16)
ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12); ax.set_zlabel('v(x,y)', fontsize=12)
ax.view_init(elev=30, azim=-45)
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
plt.savefig(os.path.join(results_dir, 'NN_solution_v_3d_styled.png'))
plt.close()
print(f"3D plot for v(x,y) saved to {os.path.join(results_dir, 'NN_solution_v_3d_styled.png')}")


