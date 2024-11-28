import numpy as np
class Topological_2D:
    def __init__(self):
        pass
    def polygon_centroid_np(self,vertices):
        """
        使用 numpy 计算多边形的质心。
        参数：
        vertices (numpy.ndarray): 形状为 (n, 2) 的 numpy 数组，其中 n 是多边形的顶点数量，每一行包含一个顶点的 x 和 y 坐标。
        返回：
        numpy.ndarray: 包含质心的 x 和 y 坐标的一维数组。
        """
        n = vertices.shape[0]
        area = 0
        centroid_x = 0
        centroid_y = 0

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            tri_area = (v1[0] * v2[1] - v2[0] * v1[1])
            area += tri_area
            centroid_x += (v1[0] + v2[0]) * tri_area
            centroid_y += (v1[1] + v2[1]) * tri_area
        area *= 0.5
        return np.array([centroid_x / (6 * area), centroid_y / (6 * area)])
    def polygon_area(self, vertices):
        """
        计算多边形的面积。
        参数：
        vertices (numpy.ndarray): 形状为 (n, 2) 的 numpy 数组，其中 n 是多边形的顶点数量，每一行包含一个顶点的 x 和 y 坐标。
        返回：
        float: 多边形的面积。
        """
        n = vertices.shape[0]
        area = 0

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            tri_area = (v1[0] * v2[1] - v2[0] * v1[1])
            area += tri_area

        area *= 0.5
        return area
    def polygon_longest_diameter(self, vertices):
        """
        计算多边形的最长直径（顶点间的最大距离）。
        参数：
        vertices (numpy.ndarray): 形状为 (n, 2) 的 numpy 数组，其中 n 是多边形的顶点数量，每一行包含一个顶点的 x 和 y 坐标。
        返回：
        float: 多边形的最长直径。
        """
        n = vertices.shape[0]
        max_distance = 0

        for i in range(n):
            for j in range(i + 1, n):
                v1 = vertices[i]
                v2 = vertices[j]
                distance = np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
                if distance > max_distance:
                    max_distance = distance

        return max_distance

