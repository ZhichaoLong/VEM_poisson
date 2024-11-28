import numpy as np

from VEM_1.Centroid import  Topological_2D



class quadrangle_mesh_2D:
    def __init__(self,domain=[0, 1, 0, 1]):
        self.domain=domain#这里是求解区域，应该是[0,1,0,1]这种形式
    def sign(self,row,ne,cnum):
         # 这里假设奇数行有3个元素，你可以根据实际需求修改这个值
        result_array = []
        current_num = cnum#起始编号
        for i in range(0, row):  # 这里假设总共生成6行数据，你可以根据实际情况修改范围
            row_r = []
            if i % 2 != 0:  # 奇数行
                for _ in range(ne):
                    row_r.append(current_num)
                    current_num += 1
            else:  # 偶数行
                for _ in range(ne - 1):
                    row_r.append(current_num)
                    current_num += 1

            result_array.append(row_r)
            result_array_np = np.array(result_array, dtype=object)
        return result_array_np
    def get_grid_edge_interior_points(self, nx, ny, n):
        """
        获取网格单元边（包括边界单元）的等分点

        :param self: 包含网格区域信息的对象（假设其有domain属性表示网格区域范围）
        :param nx: x方向的网格数量
        :param ny: y方向的网格数量
        :param n: 每条边的等分数（不包括顶点）
        :return: 按照先按y坐标排序，然后按x坐标排序后的等分点坐标数组
        """
        # 初始化一个空的列表来存储所有等分点（包括顶点）
        edge_points = []

        # 计算水平和垂直方向的步长
        hx = (self.domain[1] - self.domain[0]) / nx
        hy = (self.domain[3] - self.domain[2]) / ny

        # 遍历每个网格单元（包括边界单元）
        for i in range(nx):
            for j in range(ny):
                # 当前网格单元的四个顶点
                x1, y1 = i * hx, j * hy
                x2, y2 = (i + 1) * hx, (j + 1) * hy

                # 水平边（不包括顶点）的等分点
                for k in range(1, n):
                    x_mid = x1 + k * (x2 - x1) / n
                    edge_points.append((x_mid, y1))  # 下边界的点
                    edge_points.append((x_mid, y2))  # 上边界的点

                # 垂直边（不包括顶点）的等分点
                for k in range(1, n):
                    y_mid = y1 + k * (y2 - y1) / n
                    edge_points.append((x1, y_mid))  # 左边界的点
                    edge_points.append((x2, y_mid))  # 右边界的点

        # 将列表转换为NumPy数组
        edge_points = np.array(edge_points)
        edge_points= np.unique(edge_points, axis=0)
        # 使用 sorted 函数进行排序，首先按 y 坐标排序，然后按 x 坐标排序
        edge_points_sorted = np.array(sorted(edge_points.tolist(), key=lambda point: (point[1], point[0])))

        return edge_points_sorted
    def rectangle(self,nx,ny,n):
        '''矩形网格，输入划分的网格数,只包含顶点'''

        NC = nx * ny
        NN = (nx + 1) * (ny + 1)
        node = np.zeros((NN, 2))
        X, Y = np.mgrid[self.domain[0]:self.domain[1]:(nx + 1) * 1j, self.domain[2]:self.domain[3]:(ny + 1) * 1j]
        node[:, 1] = X.flatten()
        node[:, 0] = Y.flatten()
        cell = np.zeros((NC, 8), dtype=np.int64)
        sign = np.arange(NN).reshape(ny + 1, nx + 1)
        cell[:, 0] = sign[0:-1, 0:-1].flatten()
        cell[:, 1] = sign[:-1, 1:].flatten()
        cell[:, 2] = sign[1:, 1:].flatten()
        cell[:, 3] = sign[1:, 0:-1].flatten()
        if n==2:
            sign_2=self.sign(2*ny+1,nx+1,sign[-1,-1]+1)
            cell[:,4]= np.array([item for sublist in sign_2[:-1:2] for item in sublist[:]])
            cell[:,5] = np.array([item for sublist in sign_2[1::2] for item in sublist[1:]])
            cell[:, 6] = np.array([item for sublist in sign_2[2::2] for item in sublist[:]])
            cell[:, 7] = np.array([item for sublist in sign_2[1::2] for item in sublist[:-1]])
            node_new=self.get_grid_edge_interior_points(nx,ny,n)
            node_re=np.vstack((node,node_new))
        elif n==0:
            return node,cell

        return node_re,cell
    def boundary_retangle(self,nx,ny,nb):#nb是每条边内部自由度个数
        NC = nx * ny
        NN = (nx + 1) * (ny + 1)
        boundary_nx=np.zeros((2,nx+nb*nx))#存上下边界
        boundary_ny=np.zeros((2,ny+ny*nb))#存左右边界
        sign = np.arange(NN).reshape(ny + 1, nx + 1).T
        boundary_nx[0,:nx]=sign[:-1,0].flatten()
        boundary_nx[1, :nx] = sign[1:, -1].flatten()
        boundary_ny[0,:ny]=sign[0,1:].flatten()
        boundary_ny[1,:ny]=sign[-1,:-1].flatten()
        if nb ==0:return boundary_nx,boundary_ny
        else:
            sign_2 = self.sign(ny*nb + ny + 1, nx*nb + 1, sign[-1, -1] + 1)

            boundary_nx[0,nx:]=np.array([sign_2[0]])
            boundary_nx[1, nx:] =np.array([sign_2[-1]])
            boundary_ny_0_ny_slice = [sublist[0] for index, sublist in enumerate(sign_2) if index % 2 == 1]
            boundary_ny[0, ny:] = np.array(boundary_ny_0_ny_slice)
            boundary_ny_0_ny_slice = [sublist[-1] for index, sublist in enumerate(sign_2) if index % 2 == 1]
            boundary_ny[1, ny:] = np.array(boundary_ny_0_ny_slice)
        return boundary_nx,boundary_ny

    def rectangle_vertices(self,node,cell):#四边形顶点
        vertices=np.zeros((cell.shape[0],4,2))#四个顶点
        vertices[:, :, 0]=node[cell[:, :4], 0]
        vertices[:, :, 1] = node[cell[:, :4], 1]
        return vertices
    def elements_top(self,vertices):
        centroid = np.zeros((vertices.shape[0], 2))  # 质心
        hD = np.zeros((vertices.shape[0], 1))  # 直径
        area = np.zeros((vertices.shape[0], 1))  # 面积\
        top=Topological_2D()
        for i in np.arange(vertices.shape[0]):
            centroid[i]=top.polygon_centroid_np(vertices[i])
            hD[i]=top.polygon_longest_diameter(vertices[i])
            area[i]=top.polygon_area(vertices[i])
        return centroid,hD,area
    def is_ear(self,polygon, i):
        """
        判断多边形的一个顶点是否构成一个'耳'（可用于形成三角形且内部无其他顶点）。
        :param polygon: 多边形顶点坐标数组，形状为 (n, 2)
        :param i: 要判断的顶点索引
        :return: 如果是'耳'则返回True，否则返回False
        """
        n = polygon.shape[0]
        p1 = polygon[(i - 1) % n]
        p2 = polygon[i]
        p3 = polygon[(i + 1) % n]

        # 计算三角形面积（使用向量叉乘的一半来计算面积）
        area_triangle = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

        for j in range(n):
            if j not in [(i - 1) % n, i, (i + 1) % n]:
                p = polygon[j]
                # 判断点p是否在三角形内部（使用向量叉乘的正负来判断）
                s1 = 0.5 * np.abs((p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1]))
                s2 = 0.5 * np.abs((p3[0] - p2[0]) * (p[1] - p2[1]) - (p[0] - p2[0]) * (p3[1] - p2[1]))
                s3 = 0.5 * np.abs((p1[0] - p3[0]) * (p[1] - p3[1]) - (p[0] - p3[0]) * (p1[1] - p3[1]))
                if np.abs(s1 + s2 + s3 - area_triangle) < 1e-10:
                    return False
        return True
    def clipping_triangulation(self,polygon):
        """
        使用耳切法对多边形进行三角剖分。
        :param polygon: 多边形顶点坐标数组，形状为 (n, 2)
        :return: 三角剖分后的三角形顶点坐标数组，形状为 (m, 3, 2)，其中m为三角形个数
        """
        triangles = []
        polygon_copy = polygon.copy()
        while polygon_copy.shape[0] > 3:
            n = polygon_copy.shape[0]
            for i in range(n):
                if self.is_ear(polygon_copy, i):
                    p1 = polygon_copy[(i - 1) % n]
                    p2 = polygon_copy[i]
                    p3 = polygon_copy[(i + 1) % n]

                    triangles.append(np.array([p1, p2, p3]))
                    polygon_copy = np.delete(polygon_copy, i, axis=0)
                    break
        if polygon_copy.shape[0] == 3:
            triangles.append(polygon_copy)
        return np.array(triangles)
    def normal_out(self,vertices_1,vertices_2):
        """给定任意两个相邻的顶点，通过向量旋转求外法方向向量"""
        # 计算两点构成的向量
        vector = np.array([vertices_2[0] - vertices_1[0], vertices_2[1] - vertices_1[1]])
        # 归一化向量
        vector_length = np.linalg.norm(vector)
        if vector_length != 0:
            vector = vector / vector_length
        # 定义顺时针旋转90度的旋转矩阵
        rotation_matrix = np.array([[0, 1], [-1, 0]])
        # 使用旋转矩阵对归一化后的向量进行旋转，得到外法向量
        outer_normal = np.dot(rotation_matrix, vector)#左乘
        return outer_normal



