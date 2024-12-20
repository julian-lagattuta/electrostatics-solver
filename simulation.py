import numpy as np
import threading
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from sympy import Matrix
from numba import jit
from objects import *
from itertools import chain, product
import random

# domain = 100
np.set_printoptions(threshold=np.inf, suppress=True)
# xx,yy,zz= np.meshgrid(np.arange(domain),np.arange(domain),np.arange(domain))

# grid = np.zeros((domain,domain,domain))+0.5
E0 = 1

class BoundaryCondition:
    def apply(self, grid, coords):
        pass


class StandardBoundaryCondition(BoundaryCondition):
    def __init__(self):
        pass

    def apply(self, grid, coords):
        a = grid
        a[0, :, :] = a[1, :, :]
        a[:, 0, :] = a[:, 1, :]
        a[:, :, 0] = a[:, :, 1]
        a[:, :, -1] = a[:, :, -2]
        a[-1, :, :] = a[-2, :, :]
        a[:, -1, :] = a[:, -2, :]


class UniformBoundaryCondition(BoundaryCondition):
    def __init__(self, top, bottom) -> None:
        self.top = top
        self.bottom = bottom

    def apply(self, grid, coords):
        grid[0, :, :] = self.bottom
        grid[-1, :, :] = self.top
        grid[:, 0, :] = grid[:, 1, :]
        grid[:, -1, :] = grid[:, -2, :]
        grid[:, :, -1] = grid[:, :, -2]
        grid[:, :, 0] = grid[:, :, 1]


@jit(nopython=True)
def iteration(grid, domain, permitivity, charge_grid):
    w = 1
    for x in range(domain):
        for y in range(domain):
            for z in range(domain):

                if 0 in (x, y, z) or domain - 1 in (x, y, z):
                    continue
                ez = (permitivity[x, y, z] + permitivity[x - 1, y, z] + permitivity[x, y - 1, z] + permitivity[
                    x - 1, y - 1, z]) / 4
                enz = (permitivity[x, y, z - 1] + permitivity[x - 1, y, z - 1] + permitivity[x, y - 1, z - 1] +
                       permitivity[x - 1, y - 1, z - 1]) / 4
                ex = (permitivity[x, y, z] + permitivity[x, y - 1, z] + permitivity[x, y, z - 1] + permitivity[
                    x, y - 1, z - 1]) / 4
                enx = (permitivity[x - 1, y, z] + permitivity[x - 1, y - 1, z] + permitivity[x - 1, y, z - 1] +
                       permitivity[x - 1, y - 1, z - 1]) / 4

                ey = (permitivity[x, y, z] + permitivity[x - 1, y, z] + permitivity[x, y, z - 1] + permitivity[
                    x - 1, y, z - 1]) / 4
                eny = (permitivity[x, y - 1, z] + permitivity[x - 1, y - 1, z] + permitivity[x, y - 1, z - 1] +
                       permitivity[x - 1, y - 1, z - 1]) / 4
                if charge_grid is not None:

                    Qinside = charge_grid[x, y, z]
                else:
                    Qinside = 0
                a0 = ez + enz + ex + enx + ey + eny
                grid[x, y, z] = grid[x, y, z] * (1 - w) + w * (
                            ex * grid[x + 1, y, z] + ey * grid[x, y + 1, z] + ez * grid[x, y, z + 1] + enx * grid[
                        x - 1, y, z] + eny * grid[x, y - 1, z] + enz * grid[x, y, z - 1] + Qinside / E0) / a0


class Simulation:
    def __init__(self, domain, boundary_condition: BoundaryCondition) -> None:
        xx, yy, zz = np.meshgrid(np.arange(domain), np.arange(domain), np.arange(domain), indexing="ij")
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.conductors: list[Sphere] = []
        self.dieletrics: list[LinearDielectricSphere] = []
        self.charges: list[ChargedSphere] = []
        self.domain = domain
        w = 1.97
        self.w = w

        self.boundary_condition = boundary_condition

    def apply_boundary_conditions(self, grid, sphere, fill):
        self.boundary_condition.apply(grid, (self.xx, self.yy, self.zz))
        # a =grid
        # a[0,:,:] = a[1,:,:]
        # a[:,0,:] = a[:,1,:]
        # a[:,:,0] = a[:,:,1]
        # a[:,:,-1] = a[:,:,-2]
        # a[-1,:,:] = a[-2,:,:]
        # a[:,-1,:] = a[:,-2,:]

        for conductor in self.conductors:
            if conductor is sphere:
                continue

            conductor.fill(grid, (self.xx, self.yy, self.zz), 0)
        if sphere is not None:
            sphere.fill(grid, (self.xx, self.yy, self.zz), fill)

    def solve_sphere(self, sphere, iterations, permitivity, prev_grid=None, charge_grid=None):

        grid = np.zeros((self.domain, self.domain, self.domain))

        # fill = float(input("input fill value")
        fill = 1
        print("WARNING")
        for i in range(iterations):
            iteration(grid, self.domain, permitivity, charge_grid)
            print(i)
            # graph(self,self.domain,np.gradient(-grid),5,30,grid)
            # grid= convolve(grid,self.convolution)
            self.apply_boundary_conditions(grid, sphere, fill)

        return grid

    def solve_charge(self, charge: ChargedSphere, iterations, permitivity):
        grid = np.zeros((self.domain, self.domain, self.domain))
        charge_grid = np.zeros_like(grid)
        charge.fill_charge(charge_grid, (self.xx, self.yy, self.zz))
        for i in range(iterations):
            iteration(grid, self.domain, permitivity, charge_grid)
            print(i)
            # grid= convolve(grid,self.convolution)
            self.apply_boundary_conditions(grid, None, 0)
        return grid

    def draw(self, total_field, zz):

        cmap = plt.get_cmap("coolwarm")
        colors = cmap(np.linspace(0, 1, 257))
        for sphere in self.conductors:
            most_charge = -99999999
            least_charge = 99999999
            for pos, normal in sphere.normals():
                if pos[2] != zz:
                    continue
                vec = (total_field[0][pos[0], pos[1], pos[2]], total_field[1][pos[0], pos[1], pos[2]],
                       total_field[2][pos[0], pos[1], pos[2]])
                charge = np.dot(normal, np.array(vec))
                most_charge = max(most_charge, charge)
                least_charge = min(least_charge, charge)
            for pos, normal in sphere.normals():
                if pos[2] != 30:
                    continue
                vec = (total_field[0][pos[0], pos[1], pos[2]], total_field[1][pos[0], pos[1], pos[2]],
                       total_field[2][pos[0], pos[1], pos[2]])
                charge = np.dot(normal, np.array(vec))
                if charge < 0:
                    m = "blue"
                    if least_charge == 0:
                        value = 1
                    else:
                        value = 1 - charge / least_charge
                else:
                    m = "red"
                    if most_charge == 0:
                        value = 1
                    else:
                        value = 1 + charge / most_charge
                value = round(value * 128)

                plt.plot([pos[0], pos[0] + normal[0] * 2], (pos[1], pos[1] + normal[1] * 2), color=m)
                plt.plot([pos[0], pos[0] + normal[0]], (pos[1], pos[1] + normal[1]), color=colors[value])
        for d in self.dieletrics:
            d.draw(zz)
        for q in self.charges:
            q.draw(zz)

    def solve(self, iterations):
        permitivity = np.full((self.domain,) * 3, E0)
        for sphere in self.dieletrics:
            sphere.fill_dielectric(permitivity, (self.xx, self.yy, self.zz))
        charge_voltage_fields = []
        charge_electric_fields = []
        for charge in self.charges:
            v = self.solve_charge(charge, iterations, permitivity)
            charge_voltage_fields.append(v)
            charge_electric_fields.append(list(np.gradient(-v)))

        if len(self.conductors) == 0:
            if len(self.charges) == 0:
                raise Exception("empty board")
            total_voltage_field = charge_voltage_fields[0]
            total_electric_field = charge_electric_fields[0]
            for i in range(1, len(charge_voltage_fields)):
                for j in range(3):
                    total_electric_field[j] += charge_electric_fields[i][j]
                total_voltage_field += charge_voltage_fields[i]
            return total_electric_field, total_voltage_field

        vector_fields = []
        voltage_fields = []
        sphere_v = None
        for sphere in self.conductors:
            sphere_v = self.solve_sphere(sphere, iterations, permitivity, sphere_v)
            sphere_e = np.gradient(-sphere_v)
            vector_fields.append(sphere_e)
            voltage_fields.append(sphere_v)

        multiplier_matrix = np.zeros((len(self.conductors), len(self.conductors) + len(self.charges)))

        for field_idx, field in enumerate(vector_fields):
            for sphere_idx, sphere in enumerate(self.conductors):
                unnormalized_charge = 0

                normal_points = 0
                for pos, normal in sphere.normals():
                    vec = (field[0][pos[0], pos[1], pos[2]], field[1][pos[0], pos[1], pos[2]],
                           field[2][pos[0], pos[1], pos[2]])

                    normal_points += 1
                    unnormalized_charge += np.dot(normal, np.array(vec)) / E0
                unnormalized_charge *= sphere.surface_area / normal_points
                print(unnormalized_charge)
                multiplier_matrix[sphere_idx, field_idx] = unnormalized_charge
        for conductor_idx, conductor in enumerate(self.conductors):
            for charge_idx, charge in enumerate(self.charges):
                unnormalized_charge = 0
                field = charge_electric_fields[charge_idx]

                normal_points = 0
                for pos, normal in conductor.normals():
                    vec = (field[0][pos[0], pos[1], pos[2]], field[1][pos[0], pos[1], pos[2]],
                           field[2][pos[0], pos[1], pos[2]])

                    unnormalized_charge += np.dot(normal, np.array(vec)) / E0
                    normal_points += 1

                unnormalized_charge *= conductor.surface_area / normal_points
                multiplier_matrix[conductor_idx, len(self.conductors) + charge_idx] = unnormalized_charge

        total_charges = np.array([x.total_charge for x in self.conductors]).reshape(-1, 1)

        augmented_matrix = Matrix(np.hstack((multiplier_matrix, total_charges)))

        rref = augmented_matrix.rref()[0]
        total_electric_field = [np.zeros((self.domain,) * 3) for _ in range(3)]
        total_voltage_field = np.zeros((self.domain,) * 3)
        print(augmented_matrix)
        print(rref)
        for conductor_idx in range(len(vector_fields)):
            multiplier = rref[conductor_idx, -1]
            for charge_idx in range(len(self.charges)):
                multiplier -= rref[conductor_idx, charge_idx + len(self.conductors)]

            multiplier = float(multiplier)
            print(multiplier)
            for i in range(3):
                total_electric_field[i] += multiplier * vector_fields[conductor_idx][i]

            total_voltage_field += multiplier * voltage_fields[conductor_idx]
        for charge_idx in range(len(self.charges)):
            for i in range(3):
                total_electric_field[i] += charge_electric_fields[charge_idx][i]
            total_voltage_field += charge_voltage_fields[charge_idx]

        field = total_electric_field
        # for sphere_idx ,sphere in enumerate(self.conductors):
        # unnormalized_charge = 0

        # normal_points = 0
        # for pos,normal in sphere.normals():

        # vec = (field[0][pos[0],pos[1],pos[2]],field[1][pos[0],pos[1],pos[2]],field[2][pos[0],pos[1],pos[2]])

        # normal_points+=1
        # unnormalized_charge+= np.dot(normal,np.array(vec))
        # charge = np.dot(normal,np.array(vec))
        # if charge==0:
        # print("zero:",pos,normal)
        # print(charge,unnormalized_charge)
        # unnormalized_charge*=sphere.surface_area/normal_points
        # print("fixed",unnormalized_charge)
        # return vector_fields[0],total_voltage_field
        return total_electric_field, total_voltage_field




