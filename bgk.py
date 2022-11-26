import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def initialize_maxwellian(m, k, T, cx, cy, cz):
    A = (m/(2 * np.pi * k * T))**1.5
    beta = -(m/(2 * k * T))
    dist = A * np.exp(beta * (cx**2 + cy**2 + cz**2))

    return dist


def calculate_moment(dist):
    mu = np.zeros(3)

    return mu


# Input variables.
numCx = 101
numCy = 101
numCz = 101
numXj = 10  # number of spatial points
Cx_l = -1800 
Cx_u = 1800
Cy_l = -1800
Cy_u = 1800
Cz_l = -1800
Cz_u = 1800

m = 6.6335209e-26  # mass of argon [kg]
k = 1.380649e-23  # Boltzmann constant [m^2 kg s^-2 K^-1]

T1 = 300  # temperature [K]
P1 = 6.6667  # pressure [Pa]
Ma1 = 9  # Mach number

T2 = 16.6927297 * T1
P2 = 94.3333333 * P1
Ma2 = 0.3898

nu = 0.1  # collision frequency (guess)
t_end = 10
h = 0.025


# Start program.
distribution_grid = np.zeros((int(t_end/h) + 1, numXj, numCx, numCy, numCz))  # initialize a Maxwellian on each spatial point
cx_vec = np.linspace(Cx_l, Cx_u, numCx, endpoint=True)
cy_vec = np.linspace(Cy_l, Cy_u, numCy, endpoint=True)
cz_vec = np.linspace(Cz_l, Cz_u, numCz, endpoint=True)

[cx, cy, cz] = np.meshgrid(cx_vec, cy_vec, cz_vec)

# Initialize Maxwellian initial conditions.
distribution_grid[:, 0, :, :, :] = initialize_maxwellian(m, k, T1, cx, cy, cz)
for point in range(1, int(numXj/2)):
    distribution_grid[0, point, :, :, :] = initialize_maxwellian(m, k, T1, cx, cy, cz)
for point in range(int(numXj/2), numXj):
    distribution_grid[0, point, :, :, :] = initialize_maxwellian(m, k, T2, cx, cy, cz)

ic_maxwellian = distribution_grid
delta_x = np.abs(cx_vec[1] - cx_vec[0])
a = 1  #TODO: Multiply by cx instead of 1.

# Time step the solution.
for t in range(0, int(t_end/h)):
    ode_list = np.zeros((numXj, numCx, numCy, numCz))

    for point in range(0, numXj):
        # TODO: Calculate the Maxwellian based on the second moment.
        source = nu * (ic_maxwellian[point, :, :, :] - distribution_grid[t, point, :, :, :])

        if (point > 0):
            diff = a * (distribution_grid[t, point, :, :, :] - distribution_grid[t, point - 1, :, :, :])/delta_x
            dfdt = -diff + source
            ode_list[point, :, :, :] = dfdt[0]

            # RK4 time.
            k1 = ode_list[point, :, :, :]
            k2 = ode_list[point, :, :, :] + (h * k1/2)
            k3 = ode_list[point, :, :, :] + (h * k2/2)
            k4 = ode_list[point, :, :, :] + (h * k3)

            distribution_grid[t + 1, point, :, :, :] = distribution_grid[t, point, :, :, :] + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * h


fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)

for t in range(0, int(t_end/h)):
    I = np.trapz(np.trapz(distribution_grid[t, 5], cz_vec, 3), cy_vec, 2)
    ax1.plot(cx_vec, I)

plt.show()
