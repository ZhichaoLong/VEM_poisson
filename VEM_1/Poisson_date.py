import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

class poisson_function:
    """
    以下是方程的具体信息，包括真解，边界条件，偏导数等

    """
    def __init__(self,c):
        self.x=sp.symbols('x')
        self.y=sp.symbols('y')
        self.c=c#u前面的系数
    def domain(self):#求解域
        return [0,1,0,1]
    def solution_sym(self):#真解符号函数
        x=self.x
        y=self.y
        u= sp.sin(sp.pi*x)*sp.sin(sp.pi * y)
        return u
    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        u_func=np.sin(np.pi*x)*np.sin(np.pi*y)
        return u_func
    def gradient_x(self,p):
        x = self.x
        y = self.y
        u_grad_x=sp.diff(self.solution_sym(),x)
        u_x=sp.lambdify((x,y),u_grad_x,modules='numpy')
        return u_x(p[...,0],p[...,1])
    def gradient_y(self, p):
        x = self.x
        y = self.y
        u_grad_y = sp.diff(self.solution_sym(), y)
        u_y = sp.lambdify((x, y), u_grad_y, modules='numpy')
        return u_y(p[..., 0], p[..., 1])
    def gradient_xx(self,p):
        x = self.x
        y = self.y
        u_grad_xx = sp.diff(self.solution_sym(), x, 2)
        u_xx = sp.lambdify((x, y), u_grad_xx, modules='numpy')
        return u_xx(p[..., 0], p[..., 1])
    def gradient_yy(self, p):
        x = self.x
        y = self.y
        u_grad_yy = sp.diff(self.solution_sym(), y, 2)
        u_yy = sp.lambdify((x, y), u_grad_yy, modules='numpy')
        return u_yy(p[..., 0], p[..., 1])
    def source(self,p):
        x = self.x
        y = self.y
        #function = -sp.diff(self.solution_sym(), x, 2)-sp.diff(self.solution_sym(), y, 2)+self.c * self.solution_sym()
        function=2*sp.pi**2*sp.sin(sp.pi * x)*sp.sin(sp.pi*y)
        source_f = sp.lambdify((x, y), function, modules='numpy')
        return source_f(p[..., 0], p[..., 1])
    def Dirichlet(self,p):
        val=self.solution(p)
        return 0
    def Neumann(self,p):
        val_x=self.gradient_x(p)
        val_y=self.gradient_y(p)
        return [val_x,val_y]

    def plot_uh(self,node,cell,uh):
        # 分离出顶点坐标和中点坐标
        vertex_coords = node[:len(node) // 2]
        midpoint_coords = node[len(node) // 2:]

        vertex_uh = uh[:len(vertex_coords)]
        midpoint_uh = uh[len(vertex_coords):]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 生成用于绘制曲面的网格数据（基于顶点坐标范围）
        x_min, x_max = np.min(vertex_coords[:, 0]), np.max(vertex_coords[:, 0])
        y_min, y_max = np.min(vertex_coords[:, 1]), np.max(vertex_coords[:, 1])
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)


        all_coords = np.vstack((vertex_coords, midpoint_coords))
        all_uh = np.hstack((vertex_uh, midpoint_uh))

        uh_grid = griddata(all_coords, all_uh, (X_grid, Y_grid), method='cubic')#将数值解网格中插值，画连续近似图

        # 为顶点坐标添加数值解作为z坐标，以便绘制散点图
        vertex_coords_with_uh = np.c_[vertex_coords, vertex_uh]#按列堆叠，形成三维坐标

        # 绘制三维曲面
        surf = ax.plot_surface(X_grid, Y_grid, uh_grid, cmap='rainbow')

        # 绘制顶点的散点（已添加数值解作为z坐标）
        ax.scatter(vertex_coords_with_uh[:, 0], vertex_coords_with_uh[:, 1], vertex_coords_with_uh[:, 2], c=vertex_uh,
                   cmap='Blues',
                   marker='o', label='Vertices')

        # 为中点坐标添加数值解作为z坐标，以便绘制散点图
        midpoint_coords_with_uh = np.c_[midpoint_coords, midpoint_uh]

        # 绘制中点的散点（已添加数值解作为z坐标）
        ax.scatter(midpoint_coords_with_uh[:, 0], midpoint_coords_with_uh[:, 1], midpoint_coords_with_uh[:, 2],
                   c=midpoint_uh,
                   cmap='rainbow',
                   marker='s', label='Midpoints')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('uh')

        plt.colorbar(surf)

        plt.legend()

        plt.show()

    def plot_uh_vertex_only(self,node, cell, uh):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 直接使用node作为顶点坐标，无需再分离中点相关内容
        vertex_coords = node

        vertex_uh = uh

        # 生成用于绘制曲面的网格数据（基于顶点坐标范围）
        x_min, x_max = np.min(vertex_coords[:, 0]), np.max(vertex_coords[:, 0])
        y_min, y_max = np.min(vertex_coords[:, 1]), np.max(vertex_coords[:, 1])
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # 将顶点数值解网格中插值，画连续近似图
        uh_grid = griddata(vertex_coords, vertex_uh, (X_grid, Y_grid), method='cubic')

        # 为顶点坐标添加数值解作为z坐标，以便绘制散点图
        vertex_coords_with_uh = np.c_[vertex_coords, vertex_uh]

        # 绘制三维曲面
        surf = ax.plot_surface(X_grid, Y_grid, uh_grid, cmap='rainbow')

        # 绘制顶点的散点（已添加数值解作为z坐标）
        ax.scatter(vertex_coords_with_uh[:, 0], vertex_coords_with_uh[:, 1], vertex_coords_with_uh[:, 2], c=vertex_uh,
                   cmap='Blues',
                   marker='o', label='Vertices')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('uh')

        plt.colorbar(surf)

        plt.legend()

        plt.show()
