import numpy as np
class Gauss_Lobatto_reference_1D:
    def __init__(self):
        pass
    def gauss_point_number(self,p):#先输入高斯积分点数量
        return p
    def gauss_coefficient_reference_1D(self,gauss_point_number):#高斯积分系数
        if gauss_point_number==3:
            gauss_coefficient_reference_1D = [1/3,4/3,1/3]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==4:
            gauss_coefficient_reference_1D = [1/6,5/6,5/6,1/6]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==5:
            gauss_coefficient_reference_1D = [0.1,0.544444444444444,0.711111111111111,0.544444444444444,0.1]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==6:
            gauss_coefficient_reference_1D = [0.0666666666666667,0.378474956297847,0.554858377035487,0.554858377035487,0.378474956297847,0.0666666666666667]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==7:
            gauss_coefficient_reference_1D = [ 0.0476190476190476,0.276826047361566,0.431745381209863,0.487619047619048,0.431745381209863,0.276826047361566,0.0476190476190476]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==8:
            gauss_coefficient_reference_1D = [0.0357142857142857,0.210704227143506,0.341122692483504,0.412458794658704,0.412458794658704,0.341122692483504,0.210704227143506,0.0357142857142857]
            return gauss_coefficient_reference_1D
        elif gauss_point_number==2:
            gauss_coefficient_reference_1D = [1,1]
            return gauss_coefficient_reference_1D
    def gauss_point_reference_1D(self,gauss_point_number):#高斯积分点
        if gauss_point_number==3:
            gauss_point_reference_1D=[-1,0,1]
            return gauss_point_reference_1D
        elif gauss_point_number==4:
            gauss_point_reference_1D=[-1, -0.447213595499958, 0.447213595499958,1]
            return gauss_point_reference_1D
        elif gauss_point_number==5:
            gauss_point_reference_1D=[-1,-0.654653670707977,0,0.654653670707977,1]
            return gauss_point_reference_1D
        elif gauss_point_number==6:
            gauss_point_reference_1D=[-1,-0.765055323929465,-0.285231516480645,0.285231516480645, 0.765055323929465,1]
            return gauss_point_reference_1D
        elif gauss_point_number==7:
            gauss_point_reference_1D=[-1,-0.830223896278567,-0.468848793470714,0,0.468848793470714,0.830223896278567,1]
            return gauss_point_reference_1D
        elif gauss_point_number==8:
            gauss_point_reference_1D=[-1, -0.871740148509607,-0.591700181433142,-0.209299217902479,0.209299217902479,0.591700181433142,0.871740148509607,1]
            return gauss_point_reference_1D
        elif gauss_point_number==2:
            gauss_point_reference_1D=[-1,1]
            return gauss_point_reference_1D
    def map_1D_to_2D_segment(self, endpoints, gauss_point_number):
        """
        将一维高斯积分点映射到二维平面上的线段

        :param endpoints: 线段的两个端点坐标，形状为 (2, 2)，例如 [[x1, y1], [x2, y2]]
        :param gauss_point_number: 一维高斯积分点数量
        :return: 二维平面上线段上的高斯积分点坐标数组和对应的积分系数数组
        """
        gauss_points_1D = self.gauss_point_reference_1D(gauss_point_number)
        gauss_coefficients_1D = self.gauss_coefficient_reference_1D(gauss_point_number)
        x1, y1 = endpoints[0]
        x2, y2 = endpoints[1]
        num_gauss_points = len(gauss_points_1D)
        gauss_points_2D = np.zeros((num_gauss_points, 2))
        # 计算线段在x和y方向上的长度变化范围
        dx = x2 - x1
        dy = y2 - y1
        for i in range(num_gauss_points):
            t = gauss_points_1D[i]
            # 对x坐标进行映射
            gauss_points_2D[i][0] = x1 + (t+1) * dx / 2
            # 对y坐标进行映射，将-1到1的区间映射到线段的y方向区间
            gauss_points_2D[i][1] = y1 + (t + 1) * dy / 2
        # 计算线段长度
        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        reference_length = 2  # 参考区间 [-1, 1] 的长度
        gauss_coefficients_1D=np.array(gauss_coefficients_1D)
        # 根据线段长度与参考区间长度的比例调整积分系数
        adjusted_gauss_coefficients_1D = gauss_coefficients_1D * segment_length / reference_length

        return gauss_points_2D, adjusted_gauss_coefficients_1D