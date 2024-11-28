import numpy as np

from VEM_1.Basis_function import Basis_function
import scipy.integrate as spi
from VEM_1.Poisson_date import poisson_function
from VEM_1.count_number import Count_number
from VEM_1.gauss_triangle import Triangle_Gauss_reference_2D
from VEM_1.quad_gauss import Gauss_reference_2D_quad, Gauss_local_quad
from VEM_1.quadrangle_mesh import quadrangle_mesh_2D



class VEM_error:
    def __init__(self,centroid,hD,area,MI,cell,node):
        self.centroid = centroid
        self.hD = hD
        self.MI=MI
        self.cell=cell
        self.node=node
        self.area=area
    def real_function(self,p,c):
        pde = poisson_function(0)  # 需要输入u的系数c
        return pde.solution(p)
    def test_function(self,p):
        return p[...,1]
    def L2_error(self,uh,TI,nk,vertices,gauss_number,c,domain,k):
        cell_num = self.cell.shape[0]#单元数
        dof_ex = self.node.shape[0]  # 边界自由度直接node维度就行
        in_c = Count_number().dim_p_k(k - 2, 2)#内部自由度
        dof=TI.shape[1]#单元自由度

        gauss = Triangle_Gauss_reference_2D()
        function = Basis_function()
        mesh = quadrangle_mesh_2D(domain)
        val=0
        for i in np.arange(cell_num):
            # 获取分块三角形的顶点：
            tri_vertices = mesh.clipping_triangulation(vertices[i])
            tri_num = tri_vertices.shape[0]  # 三角形个数
            for tr in np.arange(tri_num):#三角形积分求和
                # 获取积分点和积分系数
                gauss_point = gauss.gauss_point_local_triangle(tri_vertices[tr], gauss_number)
                gauss_weight = gauss.gauss_coefficient_local_triangle(tri_vertices[tr], gauss_number)
                for g in np.arange(gauss_weight.shape[0]):#积分点循环
                    result = 0
                    for j in np.arange(dof):
                        if j <self.cell[i].shape[0]:
                            for n in np.arange(nk):
                                result+=(( uh[self.cell[i,j]]*TI [n,j]*function.B_function(n,self.centroid[i],self.hD[i],gauss_point[g],self.MI)))

                        else:
                            num_F_in_p = dof_ex - 1 + i * in_c + j - self.cell[i].shape[0] +1#内部自由度到全局自由度编号的映射
                            for n in np.arange(nk):
                                result += ( ( uh[num_F_in_p]*TI[n, j]* function.B_function(n, self.centroid[i], self.hD[i], gauss_point[g], self.MI)))
                    u_val=self.real_function(gauss_point[g], c)
                    val+=gauss_weight[g]*(u_val-result)**2
        return np.sqrt(val)

    def L_wuqiong(self,uh,TI,nk,vertices,gauss_number,c,domain,k):
        cell_num = vertices.shape[0]#单元数
        dof_ex = self.node.shape[0]  # 边界自由度直接uh的维度就行
        in_c = Count_number().dim_p_k(k - 2, 2)
        dof=TI.shape[1]#单元自由度
        gauss = Triangle_Gauss_reference_2D()
        function = Basis_function()
        mesh = quadrangle_mesh_2D(domain)
        result=np.zeros((uh.shape[0],1))
        for i in np.arange(cell_num):
            for j in np.arange(dof):
                if j <self.cell[i].shape[0]:

                    u_val = self.real_function(self.node[self.cell[i,j]], c)
                    result[self.cell[i,j]]=np.abs(u_val-uh[self.cell[i,j]])
                else:
                    num_F_in_p = dof_ex - 1 + i * in_c + j - self.cell[i].shape[0] + 1
                    result[num_F_in_p]=0

        return np.max(result)
