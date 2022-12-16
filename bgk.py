import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def animate(i):
    ax1.lines.pop(0)
    ax1.plot(np.linspace(Xj_l, Xj_u, numXj), temperature_list[i, :], color='black')


def initialize_maxwellian(m_hat, n_hat, T_hat, v_hat, cx, cy, cz):
    A = (m_hat/(np.pi * T_hat))**1.5
    beta = -(m_hat/(T_hat))
    dist = n_hat * A * np.exp(beta * ((cx - v_hat)**2 + cy**2 + cz**2))

    return dist


def calculate_moment(dist, cx, cy, cz, cx_vec, cy_vec, cz_vec):
    mu = np.zeros(3)

    tmp1 = np.trapz(cz_vec, np.trapz(cy_vec, dist, 1), 2)
    mu[0] = -np.trapz(cx_vec, tmp1, 0)

    uk = cx * dist

    tmp2 = np.trapz(cz_vec, np.trapz(cy_vec, uk, 1), 2)
    mu[1] = -np.trapz(cx_vec, tmp2, 0)

    c2 = (cx - mu[1]/mu[0])**2 + cy**2 + cz**2
    ek = c2 * dist

    tmp3 = np.trapz(cz_vec, np.trapz(cy_vec, ek, 1), 2)
    mu[2] = -np.trapz(cx_vec, tmp3, 0)

    return mu


# Input variables.
numCx = 51
numCy = 51
numCz = 51
numXj = 100  # number of spatial points
Cx_l = -20
Cx_u = 20
Cy_l = -20
Cy_u = 20
Cz_l = -20
Cz_u = 20
Xj_l = -10
Xj_u = 10

m = 6.6335209e-26  # mass of argon [kg]
k = 1.380649e-23  # Boltzmann constant [m^2 kg s^-2 K^-1]
R = 208.13  # gas constant for argon
gamma = 1.667

T1 = 300  # temperature [K]
P1 = 6.6667  # pressure [Pa]
Ma1 = 9  # Mach number
a1 = np.sqrt(gamma * R * T1)
u1 = Ma1 * a1
rho1 = P1/(R * T1)

T2 = T1 * (((gamma - 1) * Ma1**2 + 2) * (2 * gamma * Ma1**2 - (gamma - 1)))/((gamma + 1)**2 * Ma1**2)
P2 = P1 * ((2 * gamma * Ma1**2) - (gamma - 1))/(gamma + 1)
rho2 = P2/(R * T2)
a2 = np.sqrt(gamma * R * T2)
u2 = u1 * rho1/rho2
Ma2 = Ma1 * u2/u1 * (T1/T2)**0.5

# Do non-dimensionalizing.
T_ref = T1
m_ref = m
c_ref = np.sqrt((2 * k * T_ref)/m_ref)
L = 0.001  # characteristic length [m]
d = 4.17e-10
sigma = np.pi * d**2
n_ref = P1/(R * T1) * 1/m
n_ref2 = P2/(R * T2) * 1/m
lam1 = 1/(n_ref * sigma)
# lam2 = 1/(n_ref2 * sigma)
Kn = lam1/L
print("Knudsen number:", Kn)
print("Mean free path:", lam1)

mu = m/2
nu_ref = sigma * ((8 * k * T1)/(np.pi * mu))**0.5 * n_ref  # collision frequency (guess)
nu2 = sigma * ((8 * k * T2)/(np.pi * mu))**0.5 * n_ref2  # collision frequency (guess)

nu = nu_ref/nu_ref

# Time step info.
t_end = 2
delta_t = 0.01


# Start program.
distribution_grid = np.zeros((int(t_end/delta_t) + 1, numXj, numCx, numCy, numCz), dtype=np.float16)  # initialize a Maxwellian on each spatial point
cx_vec = np.linspace(Cx_l, Cx_u, numCx, endpoint=True)
cy_vec = np.linspace(Cy_l, Cy_u, numCy, endpoint=True)
cz_vec = np.linspace(Cz_l, Cz_u, numCz, endpoint=True)
xj_vec = np.linspace(Xj_l, Xj_u, numXj, endpoint=True)

delta_x = np.abs(xj_vec[1] - xj_vec[0])
print('Delta x:', delta_x)
print('Max CFL number:', Cx_u * delta_t/delta_x)

[cx, cy, cz] = np.meshgrid(cx_vec, cy_vec, cz_vec)

# Initialize Maxwellian initial conditions.
logistic_f = np.exp(xj_vec)/(np.exp(xj_vec) + 1)
T_val = logistic_f * (T2/T_ref - T1/T_ref) + T1/T_ref
n_val = logistic_f * (n_ref2/n_ref - n_ref/n_ref) + n_ref/n_ref
u_val = logistic_f * (u2/c_ref - u1/c_ref) + u1/c_ref

for point in range(0, numXj):
    distribution_grid[0, point, :, :, :] = initialize_maxwellian(m/m_ref, n_val[point], T_val[point], u_val[point], cx, cy, cz)


# moment = calculate_moment(distribution_grid[0, numXj - 1, :, :, :], cx, cy, cz, cx_vec, cy_vec, cz_vec)
# temperature = 2/3 * moment[2]/moment[0]
# velocity = moment[1]/moment[0]
# n = moment[0]

# print(n_ref2/n_ref)
# print(n, velocity, temperature, T2/T1, rho2/rho1, u1/c_ref)

# Plot distribution marginalized on cy and cz.
# I = np.trapz(np.trapz(distribution_grid[0, 150], cz_vec, 2), cy_vec, 1)
# I2 = np.trapz(np.trapz(distribution_grid[0, 1], cz_vec, 2), cy_vec, 1)
# plt.plot(cx_vec, I2)
# plt.plot(cx_vec - u2/c_ref, I)
# plt.show()

# Time step the solution.
for t in range(0, int(t_end/delta_t)):
    for point in range(0, numXj):
        # Calculate Maxwellian based on current distribution's temperature and velocity.
        moment = calculate_moment(distribution_grid[t, point, :, :, :], cx, cy, cz, cx_vec, cy_vec, cz_vec)
        temperature = 2/3 * moment[2]/moment[0]
        velocity = moment[1]/moment[0]
        n = moment[0]
        maxwellian = initialize_maxwellian(m/m_ref, n, temperature, velocity, cx, cy, cz)

        fm = maxwellian[:, :, :]
        f = distribution_grid[t, point, :, :, :]
        Q_coll = delta_t * 1/Kn * nu * (fm - f)

        if point == numXj - 1:
            # Outlet boundary condition.
            dfdx_pos_cx = (distribution_grid[t, point, int(numCx/2):, :, :] - distribution_grid[t, point - 1, int(numCx/2):, :, :])/(delta_x)
            conv_pos_cx = delta_t * cy[int(numCx/2):, :, :] * dfdx_pos_cx

            distribution_grid[t + 1, point, 0:int(numCx/2), :, :] = distribution_grid[t + 1, point - 1, 0:int(numCx/2), :, :]
            distribution_grid[t + 1, point, int(numCx/2):, :, :] = distribution_grid[t, point, int(numCx/2):, :, :] - conv_pos_cx + Q_coll[int(numCx/2):, :, :]
        elif point == 0:
            # Inlet boundary condition.
            dfdx_neg_cx = (distribution_grid[t, point + 1, 0:int(numCx/2), :, :] - distribution_grid[t, point, 0:int(numCx/2), :, :])/(delta_x)
            conv_neg_cx = delta_t * cy[0:int(numCx/2), :, :] * dfdx_neg_cx

            distribution_grid[t + 1, point, 0:int(numCx/2), :, :] = distribution_grid[t, point, 0:int(numCx/2), :, :] - conv_neg_cx + Q_coll[0:int(numCx/2), :, :]
            distribution_grid[t + 1, point, int(numCx/2):, :, :] = distribution_grid[t, point, int(numCx/2):, :, :]
        else:
            # All other grid points.
            dfdx_neg_cx = (distribution_grid[t, point + 1, 0:int(numCx/2), :, :] - distribution_grid[t, point, 0:int(numCx/2), :, :])/(delta_x)
            dfdx_pos_cx = (distribution_grid[t, point, int(numCx/2):, :, :] - distribution_grid[t, point - 1, int(numCx/2):, :, :])/(delta_x)
            conv_neg_cx = delta_t * cy[0:int(numCx/2), :, :] * dfdx_neg_cx
            conv_pos_cx = delta_t * cy[int(numCx/2):, :, :] * dfdx_pos_cx

            distribution_grid[t + 1, point, 0:int(numCx/2), :, :] = distribution_grid[t, point, 0:int(numCx/2), :, :] - conv_neg_cx + Q_coll[0:int(numCx/2), :, :]
            distribution_grid[t + 1, point, int(numCx/2):, :, :] = distribution_grid[t, point, int(numCx/2):, :, :] - conv_pos_cx + Q_coll[int(numCx/2):, :, :]

        # Outflow boundary condition if cx is positive, don't do anything. If cx is negative, need a boundary condition. Vice versa for the inlet boundary.
        # distribution_grid[t + 1, numXj - 1, :, :, :] = distribution_grid[t + 1, numXj - 2, :, :, :]
    print (t * delta_t)


plt.rc('font', family='serif')
fig = plt.figure(figsize=(8, 8))
# fig2 = plt.figure(figsize=(8, 8))
# fig3 = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax3 = fig3.add_subplot(111)

temperature_list = np.zeros((int(t_end/delta_t), numXj), dtype=np.float16)
# velocity_list = np.zeros((int(t_end/delta_t), numXj)) 
# density_list = np.zeros((int(t_end/delta_t), numXj))
for t in range(0, int(t_end/delta_t)):
    for point in range(0, numXj):
        moment = calculate_moment(distribution_grid[t, point, :, :, :], cx, cy, cz, cx_vec, cy_vec, cz_vec)
        temperature_list[t, point] = 2/3 * moment[2]/moment[0]
        # velocity_list[t, point] = moment[1]/moment[0]
        # density_list[t, point] = moment[0]

ax1.plot(np.linspace(Xj_l, Xj_u, numXj), temperature_list[0, :], color='black')

ax1.set_xlabel('x')
ax1.set_ylabel('Temperature')
ax1.grid()
# ax2.set_xlabel('x')
# ax2.set_ylabel('Density')

ani = FuncAnimation(fig, animate, frames=int(t_end/delta_t), interval=200)
writervideo = animation.FFMpegWriter(fps=15)
ani.save('temperature.mp4', writer=writervideo)

plt.close()

# ax2.plot(np.linspace(Xj_l, Xj_u, numXj), density_list[0, :], color='black')
# ax2.plot(np.linspace(Xj_l, Xj_u, numXj), density_list[int(t_end/delta_t) - 1, :], '--', color='black')

plt.show()
