"""
A minimal ODE (Ordinary Differential Equations) model for one 'pixel' of the Living LED Matrix.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
params = {
    "k_elec_on": 1.0,    # mediator oxidation rate when electrode ON
    "k_elec_off": 0.0,   # when electrode OFF

    #TODO: check papers for all of these values. And additional equations.
    "k_red": 0.2,        # mediator reduction rate 
    "k_on": 0.3,         # SoxR activation rate by mediator
    "k_off": 0.05,       # SoxR deactivation rate
    "k_prod": 0.8,       # Lux protein production rate when active
    "k_deg": 0.1,        # Lux protein degradation/dilution rate
    "k_ph": 1.0          # light emission scaling factor
}

# ---------------------------------------------------------------------
# Electrode pulse pattern
# TODO: Square wave pattern to be added later.
# ---------------------------------------------------------------------
def electrode_flux(t, params):
    # ON between 20-40 s, 80-100 s
    if 20 <= t <= 40:
        return params["k_elec_on"]
    return params["k_elec_off"]

# ---------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------
def model(t, y, p):
    M, A, L = y  # mediator, activation, Lux
    dM = electrode_flux(t, p) - p["k_red"] * M
    dA = p["k_on"] * M * (1 - A) - p["k_off"] * A
    dL = p["k_prod"] * A - p["k_deg"] * L
    return [dM, dA, dL]

# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
t0, tf = 0, 150
y0 = [0, 0, 0]  # initial conditions
ts = np.linspace(t0, tf, 1000)
sol = solve_ivp(lambda t, y: model(t, y, params), [t0, tf], y0, t_eval=ts)

M, A, L = sol.y
Light = params["k_ph"] * L

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
ax[0].plot(ts, M, label='Mediator M')
ax[1].plot(ts, A, label='Activation A', color='orange')
ax[2].plot(ts, Light, label='Light output', color='purple')

ax[0].set_ylabel("Mediator")
ax[1].set_ylabel("SoxR activation")
ax[2].set_ylabel("Light (a.u.)")
ax[2].set_xlabel("Time (m)")
for a in ax: a.legend()
plt.suptitle("Minimal Lumped ODE Model — One Living Pixel")
plt.tight_layout()
plt.show()
