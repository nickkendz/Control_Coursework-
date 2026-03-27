"""
ELE2038 - Signals and Control: Coursework Model
=================================================
State-Space Representation and Linearisation of the
wooden ball on inclined plane with electromagnet system.

States:
  x1 = x       (ball position along incline, m)
  x2 = dx/dt   (ball velocity, m/s)
  x3 = i       (coil current, A)
  x4 = x_meas  (sensor output, m)

Input:
  u  = V       (voltage applied to coil, V)

Output:
  y  = x_meas  (measured position from sensor)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import StateSpace
import matplotlib.pyplot as plt

# ==============================================================
# 1. SYSTEM PARAMETERS
# ==============================================================
m     = 0.462        # ball mass, kg
g     = 9.81         # gravitational acceleration, m/s^2
d     = 0.42         # spring natural length position, m
delta = 0.65         # electromagnet position, m
r     = 0.123        # ball radius, m
R     = 2200.0       # coil resistance, Ohm  (2.2 kOhm)
L0    = 0.125        # nominal inductance, H  (125 mH)
L1    = 0.0241       # inductance parameter, H (24.1 mH)
alpha = 1.2          # inductance decay rate, 1/m
c     = 6.811e-3     # magnetic force constant, N*m^2/A^2
                     # (6.811 m^3 g / (A^2 s^2), g = grams -> * 1e-3 for kg)
k     = 1885.0       # spring stiffness, N/m
b     = 10.4         # viscous damping coefficient, N*s/m
phi   = np.radians(41)  # incline angle, rad
tau_m = 0.030        # sensor time constant, s (30 ms)

# Effective mass (ball rolls without sliding: I = 2/5 m r^2)
M = 1.4 * m          # = m + I/r^2 = m + (2/5)m = 7/5 m

# Equilibrium position (given)
x_eq = 0.4  # m

# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def L_func(x):
    """Position-dependent inductance L(x)."""
    y = delta - x
    return L0 + L1 * np.exp(-alpha * y)


def F_mag(i, x):
    """Magnitude of the electromagnetic force on the ball."""
    y = delta - x
    return c * i**2 / y**2


# ==============================================================
# 3. EQUILIBRIUM COMPUTATION
# ==============================================================
# From dx2/dt = 0 at equilibrium (Document 1 formulation):
#
#   0 = c * i_eq^2 / (delta - x_eq)^2  +  m*g*sin(phi)  -  k*(x_eq - d)
#
# Solve for i_eq:
#   c * i_eq^2 / (delta - x_eq)^2 = k*(x_eq - d) - m*g*sin(phi)

y_eq = delta - x_eq
spring_force_eq = k * (x_eq - d)
gravity_force_eq = m * g * np.sin(phi)

# The right-hand side that i_eq^2 must satisfy
rhs = spring_force_eq - gravity_force_eq  # = k*(x_eq - d) - mg*sin(phi)

print("=" * 60)
print("EQUILIBRIUM ANALYSIS")
print("=" * 60)
print(f"  y_eq = delta - x_eq       = {y_eq:.4f} m")
print(f"  Spring force  k*(x_eq-d)  = {spring_force_eq:.4f} N")
print(f"  Gravity force mg*sin(phi) = {gravity_force_eq:.4f} N")
print(f"  RHS = k*(x_eq-d) - mg*sin(phi) = {rhs:.4f}")

i_eq_squared = rhs * y_eq**2 / c

if i_eq_squared < 0:
    print(f"\n  WARNING: i_eq^2 = {i_eq_squared:.4f} < 0")
    print("  The equilibrium current is imaginary with the current sign convention.")
    print("  This suggests the magnetic force should oppose the spring+gravity forces.")
    print("  Adjusting sign: using |i_eq^2| (magnetic force acts in -x direction).\n")
    # Physical interpretation: the electromagnet pulls the ball TOWARD it,
    # which may be in the -x direction depending on coordinate choice.
    # We take the absolute value to find the required current magnitude.
    i_eq_squared = abs(rhs) * y_eq**2 / c

i_eq = np.sqrt(i_eq_squared)
V_eq = i_eq * R        # From dx3/dt = 0: V_eq = i_eq * R
L_eq = L_func(x_eq)    # Inductance at equilibrium

print(f"\n  Equilibrium values:")
print(f"    x_eq   = {x_eq:.4f} m")
print(f"    v_eq   = 0 m/s")
print(f"    i_eq   = {i_eq:.6f} A")
print(f"    V_eq   = {V_eq:.4f} V")
print(f"    L_eq   = {L_eq*1000:.4f} mH")

# ==============================================================
# 4. LINEARISATION — JACOBIAN / PARTIAL DERIVATIVES
# ==============================================================
# Nonlinear magnetic force term: f(x, i) = i^2 / (delta - x)^2
#
# Partial derivatives evaluated at equilibrium:
#   Kx = df/dx |_eq = 2*i_eq^2 / (delta - x_eq)^3
#   Ki = df/di |_eq = 2*i_eq   / (delta - x_eq)^2

Kx = 2 * i_eq**2 / y_eq**3
Ki = 2 * i_eq / y_eq**2

print(f"\n  Linearisation constants:")
print(f"    Kx = 2*i_eq^2 / y_eq^3 = {Kx:.6f}")
print(f"    Ki = 2*i_eq   / y_eq^2 = {Ki:.6f}")

# ==============================================================
# 5. STATE-SPACE MATRICES (4 states, including sensor)
# ==============================================================
#
# State vector:  [Δx, Δv, Δi, Δx_meas]^T
# Input:         Δu = ΔV
# Output:        y  = Δx_meas
#
# NOTE on magnetic force sign:
# The A(2,1) entry depends on whether the magnetic force is in the +x or -x
# direction. Document 1 uses +c*Kx; if the magnet opposes the spring, use -c*Kx.
# We provide BOTH options below. Choose the one matching your derivation.

# --- Option A: Magnetic force in +x direction (as in Document 1 notes) ---
A_optA = np.array([
    [0,            1,           0,          0        ],
    [(c*Kx - k)/M, -b/M,        c*Ki/M,    0        ],
    [0,            0,           -R/L_eq,    0        ],
    [1/tau_m,      0,           0,          -1/tau_m ]
])

# --- Option B: Magnetic force in -x direction (for equilibrium to exist) ---
A_optB = np.array([
    [0,             1,           0,          0        ],
    [(-c*Kx - k)/M, -b/M,       -c*Ki/M,   0        ],
    [0,             0,           -R/L_eq,    0        ],
    [1/tau_m,       0,           0,          -1/tau_m ]
])

B = np.array([
    [0],
    [0],
    [1/L_eq],
    [0]
])

C = np.array([[0, 0, 0, 1]])  # output = measured position

D = np.array([[0]])

# Select which formulation to use (change this based on your derivation)
# Using Option A (Document 1 formulation) as default:
A = A_optA

print("\n" + "=" * 60)
print("STATE-SPACE MATRICES (Option A: F_mag in +x direction)")
print("=" * 60)
print(f"\nA matrix (4x4):")
print(np.array2string(A, precision=6, suppress_small=True))
print(f"\nB matrix (4x1):")
print(np.array2string(B, precision=6, suppress_small=True))
print(f"\nC matrix (1x4):")
print(np.array2string(C, precision=6, suppress_small=True))
print(f"\nD matrix (1x1):")
print(np.array2string(D, precision=6, suppress_small=True))

# Also print Option B
print("\n" + "=" * 60)
print("STATE-SPACE MATRICES (Option B: F_mag in -x direction)")
print("=" * 60)
print(f"\nA matrix (4x4):")
print(np.array2string(A_optB, precision=6, suppress_small=True))

# ==============================================================
# 6. SYSTEM ANALYSIS
# ==============================================================
print("\n" + "=" * 60)
print("SYSTEM ANALYSIS")
print("=" * 60)

for label, A_mat in [("Option A (+x)", A_optA), ("Option B (-x)", A_optB)]:
    eigvals = np.linalg.eigvals(A_mat)
    print(f"\n--- {label} ---")
    print(f"  Eigenvalues of A:")
    for j, ev in enumerate(eigvals):
        stability = "stable" if ev.real < 0 else "UNSTABLE"
        print(f"    λ_{j+1} = {ev.real:>12.4f} {'+' if ev.imag >= 0 else '-'} {abs(ev.imag):>10.4f}j  [{stability}]")

    # Open-loop stability
    if all(ev.real < 0 for ev in eigvals):
        print(f"  => Open-loop system is STABLE")
    else:
        print(f"  => Open-loop system is UNSTABLE (needs controller)")

# Controllability (standard Kalman test)
print(f"\n--- Controllability (using Option A) ---")
n = A.shape[0]
ctrb_matrix = B.copy()
An = np.eye(n)
for j in range(1, n):
    An = An @ A
    ctrb_matrix = np.hstack([ctrb_matrix, An @ B])
sv = np.linalg.svd(ctrb_matrix, compute_uv=False)
tol = max(ctrb_matrix.shape) * sv[0] * np.finfo(float).eps
rank_ctrb = np.sum(sv > tol)
print(f"  Controllability matrix singular values: {sv}")
print(f"  Kalman rank = {rank_ctrb} / {n}")
if rank_ctrb < n:
    print(f"  NOTE: Large eigenvalue spread causes numerical ill-conditioning.")
    print(f"        Using PBH (Popov–Belevitch–Hautus) test instead...")

# PBH controllability test: rank([λI - A, B]) = n for each eigenvalue λ
eigvals_check = np.linalg.eigvals(A)
pbh_controllable = True
for ev in eigvals_check:
    pbh_mat = np.hstack([ev * np.eye(n) - A, B])
    pbh_rank = np.linalg.matrix_rank(pbh_mat)
    if pbh_rank < n:
        pbh_controllable = False
        print(f"  PBH FAIL at λ = {ev:.4f}: rank = {pbh_rank}")
if pbh_controllable:
    print(f"  PBH test: CONTROLLABLE (all eigenvalues pass)")
    rank_ctrb = n  # override for summary
else:
    print(f"  PBH test: NOT CONTROLLABLE")

# Observability (standard + PBH)
print(f"\n--- Observability (using Option A) ---")
obsv_matrix = C.copy()
CA = C.copy()
for j in range(1, n):
    CA = CA @ A
    obsv_matrix = np.vstack([obsv_matrix, CA])
sv_o = np.linalg.svd(obsv_matrix, compute_uv=False)
tol_o = max(obsv_matrix.shape) * sv_o[0] * np.finfo(float).eps
rank_obsv = np.sum(sv_o > tol_o)
print(f"  Observability matrix singular values: {sv_o}")
print(f"  Kalman rank = {rank_obsv} / {n}")

# PBH observability: rank([λI - A; C]) = n
pbh_observable = True
for ev in eigvals_check:
    pbh_mat_o = np.vstack([ev * np.eye(n) - A, C])
    pbh_rank_o = np.linalg.matrix_rank(pbh_mat_o)
    if pbh_rank_o < n:
        pbh_observable = False
        print(f"  PBH FAIL at λ = {ev:.4f}: rank = {pbh_rank_o}")
if pbh_observable:
    print(f"  PBH test: OBSERVABLE (all eigenvalues pass)")
    rank_obsv = n
else:
    print(f"  PBH test: NOT OBSERVABLE")

# ==============================================================
# 7. TRANSFER FUNCTION (from StateSpace)
# ==============================================================
print("\n" + "=" * 60)
print("TRANSFER FUNCTION")
print("=" * 60)
try:
    sys_ss = StateSpace(A, B, C, D)
    # Convert to transfer function
    from scipy.signal import ss2tf
    num, den = ss2tf(A, B, C, D)
    print(f"  Numerator coefficients:   {num[0]}")
    print(f"  Denominator coefficients: {den}")
except Exception as e:
    print(f"  Could not compute transfer function: {e}")

# ==============================================================
# 8. NONLINEAR SYSTEM SIMULATION
# ==============================================================

def nonlinear_dynamics(t, state, V_input):
    """
    Nonlinear ODEs for the ball-electromagnet system.
    state = [x, v, i, x_meas]
    V_input = applied voltage (scalar or function of t)
    """
    x1, x2, x3, x4 = state

    # Applied voltage
    V = V_input(t) if callable(V_input) else V_input

    # Distance to electromagnet
    y = delta - x1

    # Safety: prevent division by zero or negative y
    if y <= 1e-6:
        y = 1e-6

    # Position-dependent inductance
    L_val = L0 + L1 * np.exp(-alpha * y)

    # Magnetic force (sign: adjust based on your derivation)
    # Using Document 1 convention: force in +x direction
    F_m = c * x3**2 / y**2

    # Equations of motion
    dx1 = x2
    dx2 = (1 / M) * (F_m + m * g * np.sin(phi) - b * x2 - k * (x1 - d))
    dx3 = (1 / L_val) * (V - x3 * R)
    dx4 = (1 / tau_m) * (x1 - x4)

    return [dx1, dx2, dx3, dx4]


def simulate_nonlinear(x0, V_func, t_span, t_eval=None):
    """Run nonlinear simulation."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    sol = solve_ivp(
        nonlinear_dynamics,
        t_span,
        x0,
        args=(V_func,),
        t_eval=t_eval,
        method='RK45',
        max_step=0.001,
        rtol=1e-8,
        atol=1e-10
    )
    return sol


def simulate_linear(dx0, dV_func, t_span, t_eval=None):
    """
    Simulate the linearised system around equilibrium.
    dx0 = initial deviation from equilibrium
    dV_func = deviation in voltage from V_eq (function of t or scalar)
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)

    def linear_dynamics(t, dx):
        dV = dV_func(t) if callable(dV_func) else dV_func
        return (A @ dx + B.flatten() * dV).tolist()

    sol = solve_ivp(
        linear_dynamics,
        t_span,
        dx0,
        t_eval=t_eval,
        method='RK45',
        max_step=0.001,
        rtol=1e-8,
        atol=1e-10
    )
    return sol


# ==============================================================
# 9. EXAMPLE SIMULATION: Step response from equilibrium
# ==============================================================
print("\n" + "=" * 60)
print("RUNNING SIMULATIONS")
print("=" * 60)

t_span = (0, 2.0)  # 2 seconds
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Initial conditions at equilibrium
x0_eq = [x_eq, 0, i_eq, x_eq]

# Small voltage step perturbation (1% of V_eq)
dV_step = 0.01 * V_eq

# --- Linear simulation: step response ---
dx0 = [0, 0, 0, 0]  # start at equilibrium
dV_func = lambda t: dV_step if t >= 0.1 else 0.0

print(f"  Simulating linear step response (ΔV = {dV_step:.4f} V at t=0.1s)...")
sol_lin = simulate_linear(dx0, dV_func, t_span, t_eval)

# --- Nonlinear simulation: same perturbation ---
V_func_nl = lambda t: V_eq + (dV_step if t >= 0.1 else 0.0)

print(f"  Simulating nonlinear step response...")
sol_nl = simulate_nonlinear(x0_eq, V_func_nl, t_span, t_eval)

# ==============================================================
# 10. PLOTTING
# ==============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle('System Response to Small Voltage Step Perturbation', fontsize=14)

# Position
axes[0].plot(sol_lin.t, sol_lin.y[0] + x_eq, 'b-', label='Linear (x)', linewidth=1.5)
axes[0].plot(sol_nl.t, sol_nl.y[0], 'r--', label='Nonlinear (x)', linewidth=1.5)
axes[0].axhline(y=x_eq, color='gray', linestyle=':', alpha=0.5, label=f'x_eq = {x_eq} m')
axes[0].set_ylabel('Position x (m)')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Velocity
axes[1].plot(sol_lin.t, sol_lin.y[1], 'b-', label='Linear (v)', linewidth=1.5)
axes[1].plot(sol_nl.t, sol_nl.y[1], 'r--', label='Nonlinear (v)', linewidth=1.5)
axes[1].set_ylabel('Velocity v (m/s)')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Current
axes[2].plot(sol_lin.t, sol_lin.y[2] + i_eq, 'b-', label='Linear (i)', linewidth=1.5)
axes[2].plot(sol_nl.t, sol_nl.y[2], 'r--', label='Nonlinear (i)', linewidth=1.5)
axes[2].axhline(y=i_eq, color='gray', linestyle=':', alpha=0.5, label=f'i_eq = {i_eq:.4f} A')
axes[2].set_ylabel('Current i (A)')
axes[2].set_xlabel('Time (s)')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/step_response.png', dpi=150, bbox_inches='tight')
print("  Step response plot saved to step_response.png")

# --- Pole-Zero Map ---
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
eigvals_A = np.linalg.eigvals(A)
ax2.scatter(eigvals_A.real, eigvals_A.imag, marker='x', s=100, c='red', linewidths=2, zorder=5)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel('Real Part')
ax2.set_ylabel('Imaginary Part')
ax2.set_title('Open-Loop Poles (Eigenvalues of A)')
ax2.grid(True, alpha=0.3)
for j, ev in enumerate(eigvals_A):
    ax2.annotate(f'λ{j+1}', (ev.real, ev.imag), textcoords="offset points",
                 xytext=(10, 5), fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/pole_map.png', dpi=150, bbox_inches='tight')
print("  Pole map saved to pole_map.png")

# --- Bode Plot ---
from scipy.signal import bode
try:
    sys_tf = StateSpace(A, B, C, D)
    w = np.logspace(-2, 5, 1000)
    w_out, mag, phase = bode(sys_tf, w=w)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig3.suptitle('Bode Plot — Open-Loop Transfer Function', fontsize=14)

    ax3a.semilogx(w_out, mag, 'b-', linewidth=1.5)
    ax3a.set_ylabel('Magnitude (dB)')
    ax3a.grid(True, which='both', alpha=0.3)

    ax3b.semilogx(w_out, phase, 'b-', linewidth=1.5)
    ax3b.set_ylabel('Phase (deg)')
    ax3b.set_xlabel('Frequency (rad/s)')
    ax3b.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/bode_plot.png', dpi=150, bbox_inches='tight')
    print("  Bode plot saved to bode_plot.png")
except Exception as e:
    print(f"  Bode plot error: {e}")

# ==============================================================
# 11. SUMMARY
# ==============================================================
print("\n" + "=" * 60)
print("SUMMARY OF KEY RESULTS")
print("=" * 60)
print(f"  Effective mass M        = {M:.4f} kg")
print(f"  Equilibrium position    = {x_eq} m")
print(f"  Equilibrium current     = {i_eq:.6f} A")
print(f"  Equilibrium voltage     = {V_eq:.4f} V")
print(f"  Inductance at eq.       = {L_eq*1000:.4f} mH")
print(f"  Kx (position gain)      = {Kx:.6f}")
print(f"  Ki (current gain)       = {Ki:.6f}")
print(f"  Open-loop eigenvalues   = {eigvals_A}")
print(f"  Controllable?           = {rank_ctrb == n}")
print(f"  Observable?             = {rank_obsv == n}")
print("\nDone!")
