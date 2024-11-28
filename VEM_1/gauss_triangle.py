import numpy as np
class Triangle_Gauss_reference_2D:
    def __init__(self):
        pass
    def gauss_coefficient_reference_triangle(self,gauss_point_number):#高斯积分系数
        if gauss_point_number==4:
            gauss_coefficient_reference_triangle=[(1-1/np.sqrt(3))/8,(1-1/np.sqrt(3))/8,(1+1/np.sqrt(3))/8,(1+1/np.sqrt(3))/8]
            return gauss_coefficient_reference_triangle

        elif gauss_point_number==3:
            gauss_coefficient_reference_triangle = [1/6,1/6,1/6]
            return gauss_coefficient_reference_triangle

        elif gauss_point_number==9:
            gauss_coefficient_reference_triangle= [64/81*(1-0)/8,100/324*(1-np.sqrt(3/5))/8,
                                                   100/324*(1-np.sqrt(3/5))/8,100/324*(1+np.sqrt(3/5))/8,
                                                   100/324*(1+np.sqrt(3/5))/8,40/81*(1-0)/8,
                                                   40/81*(1-0)/8,40/81*(1-np.sqrt(3/5))/8,
                                                   40/81*(1+np.sqrt(3/5))/8]
            return gauss_coefficient_reference_triangle
    def gauss_point_reference_triangle(self,gauss_point_number):#高斯积分点

        if gauss_point_number==4:
            gauss_point_reference_triangle=[[(1/np.sqrt(3)+1)/2,(1-1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
                                            [(1/np.sqrt(3)+1)/2,(1-1/np.sqrt(3))*(1-1/np.sqrt(3))/4],
                                            [(-1/np.sqrt(3)+1)/2,(1+1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
                                            [(-1/np.sqrt(3)+1)/2,(1+1/np.sqrt(3))*(1-1/np.sqrt(3))/4]]
            return gauss_point_reference_triangle

        elif gauss_point_number==3:
            gauss_point_reference_triangle=[[1/2,0],[1/2,1/2],[0,1/2]]
            return gauss_point_reference_triangle

        elif gauss_point_number==9:
            gauss_point_reference_triangle=[[(1+0)/2,(1-0)*(1+0)/4],
                                            [(1+np.sqrt(3/5))/2,(1-np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                            [(1+np.sqrt(3/5))/2,(1-np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                            [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1+np.sqrt(3/5))/4],
                                            [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1-np.sqrt(3/5))/4],
                                            [(1+0)/2,(1-0)*(1+np.sqrt(3/5))/4],
                                            [(1+0)/2,(1-0)*(1-np.sqrt(3/5))/4],
                                            [(1+np.sqrt(3/5))/2,(1-np.sqrt(3/5))*(1+0)/4],
                                            [(1-np.sqrt(3/5))/2,(1+np.sqrt(3/5))*(1+0)/4]]
            return gauss_point_reference_triangle
    def gauss_coefficient_local_triangle(self, triangle_vertices,gauss_point_number):
        """
        获取给定三角形的局部高斯积分系数。
        :param triangle_vertices: numpy数组，形状为(3, 2)，存储了三角形的顶点坐标
        :return: 局部高斯积分系数数组
        """
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]

        J = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        gauss_coefficient_reference_triangle = self.gauss_coefficient_reference_triangle(gauss_point_number)
        gauss_coefficient_reference_triangle = np.array(gauss_coefficient_reference_triangle)
        gauss_coefficient_local_triangle = gauss_coefficient_reference_triangle * J
        return gauss_coefficient_local_triangle
    def gauss_point_local_triangle(self, triangle_vertices,gauss_point_number):
        """
        获取给定三角形的局部高斯积分点。
        :param triangle_vertices: numpy数组，形状为(3, 2)，存储了三角形的顶点坐标
        :return: 局部高斯积分点数组，形状为(n, 2)，其中n为积分点数量
        """
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]
        gauss_point_reference_triangle = self.gauss_point_reference_triangle(gauss_point_number)
        gauss_point_reference_triangle = np.array(gauss_point_reference_triangle)
        gauss_point_local_triangle = np.zeros((gauss_point_reference_triangle.shape))

        gauss_point_local_triangle[:, 0] = x1 + (x2 - x1) * gauss_point_reference_triangle[:, 0] + (
                    x3 - x1) * gauss_point_reference_triangle[:, 1]
        gauss_point_local_triangle[:, 1] = y1 + (y2 - y1) * gauss_point_reference_triangle[:, 0] + (
                    y3 - y1) * gauss_point_reference_triangle[:, 1]
        return gauss_point_local_triangle

