import numpy as np
import matplotlib.pyplot as plt

class Sphere:
    def __init__(self, pos, radius, total_charge):
        self.is_dielectric = False
        self.pos = pos
        self.radius = radius
        if type(radius) != int or type(self.pos[0]) != int:
            raise "not int"
        self.total_charge = total_charge
        self.memoized = None
        self.memoized_normals = []
        self.surface_area = 4 * np.pi * self.radius * self.radius

    def fill(self, grid, coords, value):
        pos = self.pos
        radius = self.radius

        if self.memoized is None:
            r = np.sqrt((pos[0] - coords[0]) ** 2 + (pos[1] - coords[1]) ** 2 + (pos[2] - coords[2]) ** 2)
            self.memoized = (r < (radius)).nonzero()
        grid[self.memoized] = value

    def normals(self):

        # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        n = round(self.surface_area)
        ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(0, n)
        theta = 2 * np.pi * i / ratio
        phi = np.arccos(1 - 2 * (i) / n)
        r = self.radius
        x, y, z = r * np.cos(theta) * np.sin(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(phi)
        coords = np.vstack((x, y, z)).T
        norms = coords / r
        coords += np.array(self.pos)
        coords = np.round(coords).astype(int)
        return zip(coords, norms)

        for z in range(-self.radius, self.radius + 1):
            mini_r = self.radius - np.abs(z)
            prev = (1,)
            for theta in np.linspace(0, 2 * np.pi - 1e-6, math.ceil(np.pi * 2 * mini_r)):

                x = round(np.cos(theta) * mini_r)
                y = round(np.sin(theta) * mini_r)
                if (x, y) == prev:
                    continue
                prev = (x, y)

                n_theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
                phi = np.arctan2(y, x)
                norm_x = np.sin(n_theta) * np.cos(phi)
                norm_y = np.sin(n_theta) * np.sin(phi)
                norm_z = np.cos(n_theta)

                v = np.array([x, y, z]) + np.array(self.pos), np.array([norm_x, norm_y, norm_z])
                v


class ConductingBox:
    def __init__(self, corner_pos_1: tuple, corner_pos_2: tuple, total_charge) -> None:
        if corner_pos_1 < corner_pos_2:
            self.corner_pos_1 = corner_pos_1
            self.corner_pos_2 = corner_pos_2
        else:
            self.corner_pos_1 = corner_pos_2
            self.corner_pos_2 = corner_pos_1

        for i in range(3):
            if self.corner_pos_1[i] == self.corner_pos_2[i]:
                raise Exception("please make box thicker than 1")
        self.total_charge = total_charge
        x1, y1, z1 = self.corner_pos_2
        x2, y2, z2 = self.corner_pos_1
        l = abs(x2 - x1)
        w = abs(y2 - y1)
        h = abs(z2 - z1)
        self.surface_area = 2 * (l * w + l * h + w * h)

        self.memoized = None

    def fill(self, grid, coords, value):
        if self.memoized is None:
            box_coords = (coords[0] >= self.corner_pos_1[0]) & (coords[1] >= self.corner_pos_1[1]) & (
                    coords[2] >= self.corner_pos_1[2]) & (coords[0] <= self.corner_pos_2[0]) & (
                                 coords[1] <= self.corner_pos_2[1]) & (coords[2] <= self.corner_pos_2[2])

            self.memoized = box_coords
        grid[self.memoized] = value

    def normals(self):

        a = self.corner_pos_1
        b = self.corner_pos_2
        x = np.linspace(a[0], b[0], b[0] - a[0] + 1)
        y = np.linspace(a[1], b[1], b[1] - a[1] + 1)
        z = np.linspace(a[2], b[2], b[2] - a[2] + 1)

        coords = np.meshgrid(x, y, z, indexing="ij")
        coords = np.column_stack((coords[0].ravel(), coords[1].ravel(), coords[2].ravel()))
        corners = set(product([a[0], b[0]], (a[1], b[1]), (a[2], b[2])))
        for i in range(3):
            h = coords[(coords[:, i] == b[i]).ravel()]
            l = coords[(coords[:, i] == a[i]).ravel()]
            norm_h = np.zeros_like(h)
            norm_h[:, i] = 1
            norm_l = -norm_h
            for p, n in zip(h, norm_h):
                if tuple(p) in corners:
                    continue
                yield p.astype(int), n

            for p, n in zip(l, norm_l):
                if tuple(p) in corners:
                    continue
                yield p.astype(int), n

        for corner in corners:

            nz = 0.69336127434
            nx = 0.69336127434
            ny = 0.69336127434
            if corner[0] == a[0]:
                nx *= -1
            if corner[1] == a[1]:
                ny *= -1
            if corner[2] == a[2]:
                nz *= -1
            yield np.array(corner).astype(int), np.array((nx, ny, nz))
        # coords[0]==a[0] or coords[1]==a[1] or coords[2] == a[2] or coords[0] == b[0] or coords[1] == b[1] or coords[2] == b[2]


class ChargedSphere:
    def __init__(self, pos, radius, total_charge):
        self.pos = pos
        self.radius = radius
        self.total_charge = total_charge
        # self.charge_density = total_charge
        self.memoized = None

    def fill_charge(self, grid, coords):
        pos = self.pos
        radius = self.radius

        if self.memoized is None:
            r = np.sqrt((pos[0] - coords[0]) ** 2 + (pos[1] - coords[1]) ** 2 + (pos[2] - coords[2]) ** 2)
            self.memoized = (r <= radius).nonzero()
        grid[self.memoized] += self.total_charge / self.memoized[0].size

    def draw(self, zz):
        pos = (self.pos[0], self.pos[1])
        if abs(self.pos[2] - zz) >= self.radius:
            return
        facecolor = "red" if self.total_charge > 0 else "blue"
        if self.total_charge == 0:
            return
        circle = plt.Circle(pos, abs(self.radius - abs(self.pos[2] - zz)), facecolor=facecolor)
        fig, ax = plt.gcf(), plt.gca()
        ax.add_patch(circle)


class ChargedBox(ConductingBox):
    def __init__(self, corner_pos_1: tuple, corner_pos_2: tuple, total_charge: float) -> None:
        super().__init__(corner_pos_1, corner_pos_2, 0)

        self.total_charge = total_charge
        vol = 1
        for i in range(3):
            vol *= self.corner_pos_2[i] - self.corner_pos_1[i]
        self.charge_density = total_charge / vol

    def fill_charge(self, grid, coords, custom_charge_density=None):
        if self.memoized is None:
            box_coords = (coords[0] >= self.corner_pos_1[0]) & (coords[1] >= self.corner_pos_1[1]) & (
                    coords[2] >= self.corner_pos_1[2]) & (coords[0] <= self.corner_pos_2[0]) & (
                                 coords[1] <= self.corner_pos_2[1]) & (coords[2] <= self.corner_pos_2[2])

            self.memoized = box_coords
        if custom_charge_density is None:
            custom_charge_density = self.charge_density
        grid[self.memoized] = custom_charge_density

    def draw(self, z):
        if self.corner_pos_1[2] <= z <= self.corner_pos_2[2]:
            facecolor = "red" if self.total_charge > 0 else "blue"
            if self.total_charge == 0:
                return
            pos = (self.corner_pos_1[0], self.corner_pos_1[1])
            rect = plt.Rectangle(pos, self.corner_pos_2[0] - self.corner_pos_1[0],
                                 self.corner_pos_2[1] - self.corner_pos_1[1], facecolor=facecolor)
            fig, ax = plt.gcf(), plt.gca()
            ax.add_patch(rect)


class LinearDielectricBox(ConductingBox):
    def __init__(self, corner_pos_1: tuple, corner_pos_2: tuple, dielectric_constant: float) -> None:
        super().__init__(corner_pos_1, corner_pos_2, 0)
        self.is_dielectric = True
        self.dielectric_constant = dielectric_constant

    def fill_dielectric(self, grid, coords):
        if self.memoized is None:
            box_coords = (coords[0] >= self.corner_pos_1[0]) & (coords[1] >= self.corner_pos_1[1]) & (
                    coords[2] >= self.corner_pos_1[2]) & (coords[0] <= self.corner_pos_2[0]) & (
                                 coords[1] <= self.corner_pos_2[1]) & (coords[2] <= self.corner_pos_2[2])

            self.memoized = box_coords
        grid[self.memoized] = self.dielectric_constant * E0

    def draw(self, z):
        if self.corner_pos_1[2] <= z <= self.corner_pos_2[2]:
            pos = (self.corner_pos_1[0], self.corner_pos_1[1])
            rect = plt.Rectangle(pos, self.corner_pos_2[0] - self.corner_pos_1[0],
                                 self.corner_pos_2[1] - self.corner_pos_1[1], edgecolor="blue", facecolor="none")
            fig, ax = plt.gcf(), plt.gca()
            ax.add_patch(rect)


class LinearDielectricSphere(Sphere):
    def __init__(self, pos, radius, dielectric_constant):
        # DOES NOT SUPPORT DIELETRICS NEAR EACH OTHER
        super().__init__(pos, radius, 0)
        self.is_dielectric = True
        self.dielectric_constant = dielectric_constant

    def draw(self, z):
        pos = (self.pos[0], self.pos[1])
        if abs(self.pos[2] - z) >= self.radius:
            return
        circle = plt.Circle(pos, abs(self.radius - abs(self.pos[2] - z)), edgecolor="blue", facecolor="none")
        fig, ax = plt.gcf(), plt.gca()
        ax.add_patch(circle)

    def fill(self, grid, coords, value):
        0 / 0

    def fill_dielectric(self, grid, coords):
        pos = self.pos
        radius = self.radius

        if self.memoized is None:
            r = np.sqrt((pos[0] - coords[0]) ** 2 + (pos[1] - coords[1]) ** 2 + (pos[2] - coords[2]) ** 2)
            self.memoized = (r <= radius).nonzero()
        grid[self.memoized] = self.dielectric_constant * E0


