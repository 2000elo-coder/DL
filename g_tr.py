import numpy as np
from sympy import symbols, diff, sin, pi, lambdify

# define symbols
x, y = symbols('x y')

# Exact solutions
u_sym = sin(pi*x)**2 * sin(pi*y)**2     #take factor of 1/2pie^2 for better loss results
v_sym = (x**2 - 1)**2 * (y**2 - 1)**2

# --- Mathematical Operators ---
def laplacian(expr, x_sym, y_sym):
    return diff(expr, x_sym, 2) + diff(expr, y_sym, 2)

def biharmonic(expr, x_sym, y_sym):
    return laplacian(laplacian(expr, x_sym, y_sym), x_sym, y_sym)

def von_karman_bracket(a_sym, b_sym, x_sym, y_sym):
    a_xx = diff(a_sym, x_sym, 2)
    a_yy = diff(a_sym, y_sym, 2)
    a_xy = diff(a_sym, x_sym, y_sym)
    b_xx = diff(b_sym, x_sym, 2)
    b_yy = diff(b_sym, y_sym, 2)
    b_xy = diff(b_sym, x_sym, y_sym)
    return a_xx * b_yy + a_yy * b_xx - 2 * a_xy * b_xy

# --- Derive PDEs and Boundary Conditions ---
biharmonic_u_gt = biharmonic(u_sym, x, y)
bracket_uv_gt = von_karman_bracket(u_sym, v_sym, x, y)
f_sym = biharmonic_u_gt - bracket_uv_gt

bracket_uu_gt = von_karman_bracket(u_sym, u_sym, x, y)
source_v_sym = -0.5 * bracket_uu_gt

def normal_derivative(expr, nx_sym, ny_sym, x_sym, y_sym):
    return nx_sym * diff(expr, x_sym) + ny_sym * diff(expr, y_sym)

# --- Lambdify all expressions for numerical evaluation ---
ld_u, ld_v, ld_f, ld_source_v = [lambdify((x, y), expr, 'numpy') for expr in [u_sym, v_sym, f_sym, source_v_sym]]
nx_sym, ny_sym = symbols('nx ny')
ld_du_dn = lambdify((x, y, nx_sym, ny_sym), normal_derivative(u_sym, nx_sym, ny_sym, x, y), 'numpy')
ld_dv_dn = lambdify((x, y, nx_sym, ny_sym), normal_derivative(v_sym, nx_sym, ny_sym, x, y), 'numpy')

def from_seq_to_array(items):
    out = [np.array(item, dtype=np.float32).reshape(-1, 1) for item in items]
    return out[0] if len(out) == 1 else out

# --- Data Generation Functions ---
def data_gen_interior(collocations):
    x_vals, y_vals = collocations[:, 0], collocations[:, 1]
    return from_seq_to_array([ld_u(x_vals, y_vals), ld_v(x_vals, y_vals), ld_f(x_vals, y_vals), ld_source_v(x_vals, y_vals)])

def data_gen_bdry(collocations, normal_vec):
    x_vals, y_vals = collocations[:, 0], collocations[:, 1]
    nx_vals, ny_vals = normal_vec[:, 0], normal_vec[:, 1]

    return from_seq_to_array([ld_u(x_vals, y_vals), ld_du_dn(x_vals, y_vals, nx_vals, ny_vals), ld_v(x_vals, y_vals), ld_dv_dn(x_vals, y_vals, nx_vals, ny_vals)])
