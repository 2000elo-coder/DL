import torch

mse_loss = torch.nn.MSELoss()

def compute_derivatives(output, x1, x2):
    dx1 = torch.autograd.grad(output.sum(), x1, create_graph=True, retain_graph=True)[0]
    dx2 = torch.autograd.grad(output.sum(), x2, create_graph=True, retain_graph=True)[0]
    dx1x1 = torch.autograd.grad(dx1.sum(), x1, create_graph=True, retain_graph=True)[0]
    dx1x2 = torch.autograd.grad(dx1.sum(), x2, create_graph=True, retain_graph=True)[0]
    dx2x2 = torch.autograd.grad(dx2.sum(), x2, create_graph=True, retain_graph=True)[0]
    return output, dx1, dx2, dx1x1, dx1x2, dx2x2

def von_karman_bracket(u_x1x1, u_x1x2, u_x2x2, v_x1x1, v_x1x2, v_x2x2):
    return u_x1x1 * v_x2x2 + u_x2x2 * v_x1x1 - 2 * u_x1x2 * v_x1x2

def laplacian(u_x1x1, u_x2x2):
    return u_x1x1 + u_x2x2

def biharmonic_operator(u_pred, x1, x2):
    _, _, _, u_x1x1, u_x1x2, u_x2x2 = compute_derivatives(u_pred, x1, x2)
    lap_u = laplacian(u_x1x1, u_x2x2)
    lap_u_x1 = torch.autograd.grad(lap_u.sum(), x1, create_graph=True, retain_graph=True)[0]
    lap_u_x2 = torch.autograd.grad(lap_u.sum(), x2, create_graph=True, retain_graph=True)[0]
    lap_u_x1x1 = torch.autograd.grad(lap_u_x1.sum(), x1, create_graph=True, retain_graph=True)[0]
    lap_u_x2x2 = torch.autograd.grad(lap_u_x2.sum(), x2, create_graph=True, retain_graph=True)[0]
    return laplacian(lap_u_x1x1, lap_u_x2x2)

def pde_residual_loss(net_u, net_v, x1, x2, f_target, source_v_target):
    u_pred = net_u(x1, x2)
    v_pred = net_v(x1, x2)
    _, _, _, u_x1x1, u_x1x2, u_x2x2 = compute_derivatives(u_pred, x1, x2)
    _, _, _, v_x1x1, v_x1x2, v_x2x2 = compute_derivatives(v_pred, x1, x2)
    biharmonic_u = biharmonic_operator(u_pred, x1, x2)
    biharmonic_v = biharmonic_operator(v_pred, x1, x2)
    bracket_uv = von_karman_bracket(u_x1x1, u_x1x2, u_x2x2, v_x1x1, v_x1x2, v_x2x2)
    bracket_uu = von_karman_bracket(u_x1x1, u_x1x2, u_x2x2, u_x1x1, u_x1x2, u_x2x2)
    R1 = biharmonic_u - bracket_uv - f_target
    R2 = biharmonic_v + 0.5 * bracket_uu
    return mse_loss(R1, torch.zeros_like(R1)) + mse_loss(R2, torch.zeros_like(R2))

def boundary_conditions(net_u, net_v, bdx1, bdx2, nx1, nx2):
    u_pred_bd = net_u(bdx1, bdx2)
    v_pred_bd = net_v(bdx1, bdx2)
    u_x1_bd = torch.autograd.grad(u_pred_bd.sum(), bdx1, create_graph=True, retain_graph=True)[0]
    u_x2_bd = torch.autograd.grad(u_pred_bd.sum(), bdx2, create_graph=True, retain_graph=True)[0]
    du_dn_pred_bd = nx1 * u_x1_bd + nx2 * u_x2_bd
    v_x1_bd = torch.autograd.grad(v_pred_bd.sum(), bdx1, create_graph=True, retain_graph=True)[0]
    v_x2_bd = torch.autograd.grad(v_pred_bd.sum(), bdx2, create_graph=True, retain_graph=True)[0]
    dv_dn_pred_bd = nx1 * v_x1_bd + nx2 * v_x2_bd
    return u_pred_bd, du_dn_pred_bd, v_pred_bd, dv_dn_pred_bd

def total_loss(net_u, net_v, intx1, intx2, f_target, source_v_target,
               bdx1, bdx2, nx1, nx2,
               g1_target, h1_target, g2_target, h2_target,
               lambda_pde, lambda_bc):
    loss_pde = pde_residual_loss(net_u, net_v, intx1, intx2, f_target, source_v_target)
    u_pred_bd, du_dn_pred_bd, v_pred_bd, dv_dn_pred_bd = boundary_conditions(net_u, net_v, bdx1, bdx2, nx1, nx2)
    loss_bc = (mse_loss(u_pred_bd, g1_target) + mse_loss(du_dn_pred_bd, h1_target) +
               mse_loss(v_pred_bd, g2_target) + mse_loss(dv_dn_pred_bd, h2_target))
    total = lambda_pde * loss_pde + lambda_bc * loss_bc
    # Return unweighted components for simplified logging
    return total, loss_pde, loss_bc