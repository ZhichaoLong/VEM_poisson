import numpy as np
class Gauss_reference_2D_quad:
    def __init__(self):
        pass
    def gauss_coefficient_reference_quad(self,gauss_point_number):#高斯积分系数
        if gauss_point_number==4:
            gauss_coefficient_reference_quad=[1,1,1,1]
            return gauss_coefficient_reference_quad
        if gauss_point_number==9:
            gauss_coefficient_reference_quad = [25/81,25/81,25/81,25/81,40/81,40/81,40/81,40/81,64/81]
            return gauss_coefficient_reference_quad

    def gauss_point_reference_quad(self, gauss_point_number):  # 高斯积分点

        if gauss_point_number == 4:
            gauss_point_reference_quad = [
                [-np.sqrt(3),-np.sqrt(3)],
                [np.sqrt(3),-np.sqrt(3)],
                [np.sqrt(3),np.sqrt(3)],
                [-np.sqrt(3),np.sqrt(3)]]
            return gauss_point_reference_quad
        if gauss_point_number == 9:
            gauss_point_reference_quad = [
                [-np.sqrt(3/5),-np.sqrt(3/5)],
                [np.sqrt(3/5),-np.sqrt(3/5)],
                [np.sqrt(3/5),np.sqrt(3/5)],
                [-np.sqrt(3/5),np.sqrt(3/5)],
                [0,-np.sqrt(3/5)],
                [np.sqrt(3/5),0],
                [0,np.sqrt(3/5)],
                [-np.sqrt(3/5),0],
                [0,0]]
            return gauss_point_reference_quad

class Gauss_local_quad:
    def __init__(self, gauss_coefficient_reference, gauss_point_reference, node, cell):
        self.gauss_coefficient_reference= gauss_coefficient_reference
        self.gauss_point_reference= gauss_point_reference
        self.node = node
        self.cell = cell

    def gauss_coefficient_local_quad(self, a,b,c,d):  # 第p个单元的局部高斯点和高斯积分系数

        gauss_coefficient_reference=self.gauss_coefficient_reference

        J=((b-a)*(d-c))/4

        gauss_coefficient_reference_quad = np.array(gauss_coefficient_reference)
        gauss_coefficient_local_quad = gauss_coefficient_reference_quad * J
        return gauss_coefficient_local_quad

    def gauss_point_local_quad(self,a,b,c,d):


        gauss_point_reference=self.gauss_point_reference
        gauss_point_reference=np.array(gauss_point_reference)
        gauss_point_local_quad=np.zeros((gauss_point_reference.shape))
        gauss_point_local_quad[:,0]=(b+a+(b-a)*gauss_point_reference[:,0])/2
        gauss_point_local_quad[:, 1] = (d + c + (d - c) * gauss_point_reference[:, 1]) / 2
        return gauss_point_local_quad