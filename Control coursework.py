import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



#System parameters
m = 0.462          # Mass (kg)
g = 9.81           # Gravity (m/s^2)
d = 0.42           # Unstretched spring distance (m)
delta = 0.65       # Inductor position (m)
R = 2200.0         # Resistance (Ohms)
L0 = 0.125         # Nominal inductance (Henries)
L1 = 0.0241        # Inductance variance (Henries)
alpha = 1.2        # Inductance constant (m^-1)
c = 6.811          # Magnetic force constant
k = 1885.0         # Spring stiffness (N/m)
b = 10.4           # Damping coefficient (Ns/m)
phi = np.radians(41) # Angle in radians (converted from 41 degrees)
tau_m = 0.030      # Sensor time constant (seconds)


#equalibrium points
x1_eq = 0.5
x2_eq = 0
x4_eq  = 0.5

#calculating x3_eq steady current sub x2_eq = 0 into the second equation
# 0 = mgsin(phi) + cx3^2/(delta - x1_eq)^2 - k(x1_eq)
 
 #set the variable to help solving

force = k*(x1_eq-d) - m*g*np.sin(phi)
x3_eq = (delta - x1_eq) * np.sqrt(force / c)

V_eq = x3_eq * R

print(f"Equilibrium Current (x3*): {x3_eq:.4f} A")
print(f"Equilibrium Voltage (V*):  {V_eq:.4f} V\n")


# Build the jacobian matrix A and B 
L_eq = L0 + L1 * np.exp(-alpha * (delta - x1_eq))

A = np.zeros((4,4))

A[0,1] = 1.0
A[1, 0] = (1 / (1.4 * m)) * ( (2 * c * x3_eq**2) / (delta - x1_eq)**3 - k )
A[1, 1] = -b / (1.4 * m)
A[1, 2] = (2 * c * x3_eq) / (1.4 * m * (delta - x1_eq)**2)

A[2, 2] = -R / L_eq
A[3, 0] = 1 / tau_m
A[3, 3] = -1 / tau_m


B = np.zeros((4, 1))

B[2, 0] = 1 / L_eq




print("A Matrix:")
print(np.round(A, 4)) # Rounded for clean viewing
print("\nB Matrix:")
print(np.round(B, 4))

C = np.array([[0,0,0,1]])
D = np.array([[0]])


# linear state space system
sys_ss = ct.ss(A, B, C, D)
# checking the stability
poles = ct.poles(sys_ss)
print("System Poles:", poles)


# checking controlability 
Co = ct.ctrb(A, B)
# If the rank is 4 (number of states), it is fully controllable
if np.linalg.matrix_rank(Co) == 4:
    print("System is fully controllable!")
else:
    print("System is NOT fully controllable.")



G = ct.ss2tf(sys_ss)
print("\nPlant Transfer Function G(s):")
print(G)



#Figure 2 grapgh
# 1. Step Response of the Open-Loop Plant (G)
# We use a short time window because the instability causes it to blow up quickly
t_open = np.linspace(0, 1.0, 500) 
t, y_open = ct.step_response(G, T=t_open)

plt.figure(figsize=(6, 5))

# Subplot 1: Time Domain Response
plt.subplot(1, 2, 1)
plt.plot(t, y_open, color='red', linewidth=2)
plt.title('Open-Loop Step Response (Unstable)')
plt.xlabel('Time (seconds)')
plt.ylabel('Output Deviation ($\Delta x$)')
plt.grid(True, linestyle='--')
plt.annotate('Exponential Divergence', xy=(0.6, y_open[-1]/2), color='red')

# Subplot 2: Pole-Zero Map
plt.subplot(1, 2, 2)
poles = ct.poles(G)
zeros = ct.zeros(G)
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', s=100, label='Poles')
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', s=100, label='Zeros')
plt.axvline(0, color='black', linewidth=1)
plt.axhline(0, color='black', linewidth=1)
plt.title('Pole-Zero Map')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.grid(True, linestyle='--')
plt.legend()

plt.xlim([-50, 50]) 
plt.ylim([-50, 50])

# Highlight the unstable pole
plt.annotate('Unstable Pole at +6.01', xy=(6.01, 0), xytext=(15, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()


#figure 3 graph 
# 1. Define the Nonlinear System Dynamics
def nonlinear_dynamics(t, x, V_input):
    x1, x2, x3, x4 = x
    
    # Inductance depends on position
    L_x = L0 + L1 * np.exp(-alpha * (delta - x1))
    
    # Differential Equations (from your system parameters)
    dx1dt = x2
    # Force balance: Gravity + Magnetic - Spring - Damping
    # Note: Using the same 1.4 factor found in your A matrix logic
    dx2dt = (1/(1.4*m)) * (m*g*np.sin(phi) + (c * x3**2)/(delta - x1)**2 - k*(x1 - d) - b*x2)
    dx3dt = (1/L_x) * (V_input - R*x3)
    dx4dt = (1/tau_m) * (x1 - x4)
    
    return [dx1dt, dx2dt, dx3dt, dx4dt]

# 2. Simulation Parameters
t_span = (0, 0.5) # Short time because it's unstable
t_eval = np.linspace(0, 0.5, 500)
delta_V = 0.1 # A small 0.1V perturbation
V_total = V_eq + delta_V

# 3. Simulate Nonlinear Model
x0 = [x1_eq, x2_eq, x3_eq, x4_eq] # Start exactly at equilibrium
sol = solve_ivp(nonlinear_dynamics, t_span, x0, args=(V_total,), t_eval=t_eval)

# 4. Simulate Linear Model (Your SS system)
# We give it the small delta_V as input
t_lin, y_lin = ct.forced_response(sys_ss, T=t_eval, U=delta_V)
# Add equilibrium back to linear deviation for comparison
y_lin_absolute = y_lin + x1_eq 

# 5. Plotting Figure 3
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], 'b-', label='Nonlinear Response (ODE)', linewidth=2)
plt.plot(t_lin, y_lin_absolute, 'r--', label='Linearized Response (SS Matrix)', linewidth=2)

plt.title('Figure 3: Linear vs. Nonlinear Open-Loop Response')
plt.xlabel('Time (seconds)')
plt.ylabel('Position $x_1$ (m)')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


#Figure 4 graph
# converting state space to the transfer function




# making the PID controller
Kp = 3500
Ki = 10000  # High Ki is the secret to getting it to settle on the red line
Kd = 300 


#Create the PID transfer function
s = ct.TransferFunction.s
C_pid = Kp + Ki/s + Kd*s


# Multiply Controller and Plant: L(s) = C(s) * G(s)
L = C_pid * G

# Close the feedback loop: T(s) = L(s) / (1 + L(s))
# The 'negative=True' implies standard negative feedback
T = ct.feedback(L, 1, sign=-1)

# Verify that the new closed-loop system is stable (all poles negative)
cl_poles = ct.poles(T)
print("\nClosed-Loop Poles:\n", cl_poles)

# Simulate a Step Response 
# (e.g., asking the system to move +0.01m from the 0.5m setpoint)
time_vector = np.linspace(0, 2, 1000) # Simulate for 2 seconds
t, y = ct.step_response(T, T=time_vector)

# Plot the results
plt.figure(figsize=(8,5))
plt.plot(t, y, label='System Response', linewidth=2)
plt.axhline(1.0, color='r', linestyle='--', label='Target Setpoint (Step = 1)')
plt.title('Figure 4: Closed-Loop Step Response with PID Control')
plt.xlabel('Time (seconds)')
plt.ylabel('Deviation from Setpoint ($\Delta x$)')
plt.grid(True)
plt.legend()
plt.show()



#Figure 5: Bode Plot of the Open-Loop Plant

plt.figure(figsize=(10, 8))

# 1. Use bode_plot without the 'Plot' argument. 
# In most versions, simply passing the system G will generate the plot.
# We also use 'initial_exp=None' or just let it auto-scale.
ct.bode_plot(G, dB=True, deg=True, omega_limits=[0.1, 10000])

# 2. Add titles and clean up the layout
# The control library creates two axes: [0] is Magnitude, [1] is Phase
fig = plt.gcf() # Get current figure
axes = fig.get_axes()

axes[0].set_title('Figure 5: Bode Plot of the Linearised Open-Loop Plant G(s)')
axes[0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1].grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



