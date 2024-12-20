import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from dataclasses import dataclass
@dataclass
class GraphState:
    active_z: int


def graph(sim, domain, grid, step, graph_state: GraphState, potential):
    xx = np.arange(domain)
    yy = np.arange(domain)
    zz = graph_state.active_z
    chopped_grid = grid[0][::step, ::step, graph_state.active_z].T, grid[1][::step, ::step, graph_state.active_z].T
    magnitude = chopped_grid[0] ** 2 + chopped_grid[1] ** 2
    f = lambda x: np.sign(x) * np.log10(1 + np.abs(x))
    # chopped_grid = f(chopped_grid[0]),f(chopped_grid[1])
    plt.pcolormesh(xx, yy, potential[:, :, zz].T, cmap="coolwarm")
    plt.quiver(xx[::step], yy[::step], chopped_grid[0], chopped_grid[1], magnitude, cmap="PiYG")
    sim.draw(grid, zz)
    plt.show()

def listen(grid, graph_state: GraphState, potential):
    while True:

        v = input("would you like to get electric field (E) or voltage(V) or set z(Z): ")
        if v.lower().strip() == "v":
            x = input("type in x coord 1 you want to sample:")
            y = input("type in y coord 1 you want to sample:")
            z = input(f"type in z coord 1 you want to sample ({zz}):")
            x2 = input("type in x coord 2 you want to sample:")
            y2 = input("type in y coord 2 you want to sample:")
            z2 = input(f"type in z coord 2 you want to sample ({zz}):")

            if z == "":
                z =  graph_state.active_z
            if z2 == "":
                z2 = graph_state.active_z
            try:
                volts = abs(
                    potential[round(float(x)), round(float(y)), z] - potential[round(float(x2)), round(float(y2)), z2])
                print(volts)
                print(potential[round(float(x)), round(float(y)), z])
            except:
                print("try again")

        elif v.lower().strip() == "e":
            x = input("type in x coord you want to sample:")
            y = input("type in y coord you want to sample:")
            z = input(f"type in z coord you want to sample ({graph_state.active_z}):")
            if z == "":
                z = graph_state.active_z
            try:
                pos = round(float(x)), round(float(y)), z
                print(f"{grid[0][pos], grid[1][pos], grid[2][pos]}")
            except:
                print("try again")
        elif v.lower().strip() == "z":
            z = input("select a z coordinate")
            try:
                graph_state.active_z = int(z)
            except:
                print("try again")


