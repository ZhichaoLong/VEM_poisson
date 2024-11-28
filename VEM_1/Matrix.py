import numpy as np

from VEM_1.Basis_function import Basis_function
from VEM_1.Poisson_date import poisson_function
from VEM_1.count_number import Count_number
from VEM_1.gauss_lobatto_1D import Gauss_Lobatto_reference_1D
from VEM_1.gauss_triangle import Triangle_Gauss_reference_2D
from VEM_1.quadrangle_mesh import quadrangle_mesh_2D
class Matrix :
    def __init__(self,domain,node,cell,centroid,hD,area):
        self.node=node
        self.cell=cell
        self.centroid=centroid
        self.hD=hD
        self.area=area
        self.domain=domain
    def to_triangle(self,i,vertices):#第i个单元拆分三角形的顶点
        mesh = quadrangle_mesh_2D(self.domain)
        to_triangle=mesh.clipping_triangulation(vertices[i])
        return to_triangle
    def delta_function(self,i,j):#这里的i的取值范围是边界节点
        if i == j:
            return 1
        else:return 0
    def gauss_cell_phi(self,i,j):#第i个单元，第j个基函数的单元积分
        if j <=self.cell[i].shape[0]-1:
            return 0
        elif j==self.cell[i].shape[0] :
            return self.area[i]
        else:return 0
    def operator_p0_phi(self,vertices,k,i,j):#这里i单元,j基函数
        N_V=vertices[i].shape[0]
        if k==1:
            val=0
            for n in np.arange(vertices[i].shape[0]):
                val+=self.delta_function(j,n)
            result=1/N_V *val
            return result
        else:
            result=1/self.area[i]*self.gauss_cell_phi(i,j)
            return result
    def operator_p0_mk(self,mk,k,i,MI,vertices,gauss_point_number):
        N_V = vertices[i].shape[0]
        function = Basis_function()
        if k==1:
            val=0
            for n in np.arange(vertices[i].shape[0]):
                val+=function.B_function(mk,self.centroid[i],self.hD[i],vertices[i,n],MI)
            result=1/N_V * val
            return result
        else:
            # 获取分块三角形的顶点：
            tri_vertices = self.to_triangle(i, vertices)  # 第i个单元的分块三角形顶点
            tri_num = tri_vertices.shape[0]  # 三角形个数
            val = 0
            gauss = Triangle_Gauss_reference_2D()
            for j in np.arange(tri_num):
                # 获取积分点和积分系数
                gauss_point = gauss.gauss_point_local_triangle(tri_vertices[j], gauss_point_number)
                gauss_weight = gauss.gauss_coefficient_local_triangle(tri_vertices[j], gauss_point_number)
                for v in np.arange(gauss_weight.shape[0]):
                    val+=gauss_weight[v]*function.B_function(mk,self.centroid[i],self.hD[i],gauss_point[v],MI)
            return val/self.area[i]
    def integration_m_m(self,nk,mk,i,MI,vertices,gauss_number):
        function = Basis_function()
        # 获取分块三角形的顶点：
        tri_vertices = self.to_triangle(i, vertices)  # 第i个单元的分块三角形顶点
        tri_num = tri_vertices.shape[0]  # 三角形个数
        val = 0
        gauss = Triangle_Gauss_reference_2D()
        for j in np.arange(tri_num):
            # 获取积分点和积分系数
            gauss_point = gauss.gauss_point_local_triangle(tri_vertices[j], gauss_number)
            gauss_weight = gauss.gauss_coefficient_local_triangle(tri_vertices[j], gauss_number)
            for v in np.arange(gauss_weight.shape[0]):  # 高斯积分，梯度的内积，就是x,y偏导乘积的和做积分
                val += (gauss_weight[v] * (
                            function.B_function(nk, self.centroid[i], self.hD[i], gauss_point[v], MI)
                            * function.B_function(mk, self.centroid[i], self.hD[i], gauss_point[v], MI)))
        return val
    def integration_f_m(self,mk,i,MI,vertices,gauss_number,c):
        pde=poisson_function(c)
        function = Basis_function()
        # 获取分块三角形的顶点：
        tri_vertices = self.to_triangle(i, vertices)  # 第i个单元的分块三角形顶点
        tri_num = tri_vertices.shape[0]  # 三角形个数
        val = 0
        gauss = Triangle_Gauss_reference_2D()
        for j in np.arange(tri_num):
            # 获取积分点和积分系数
            gauss_point = gauss.gauss_point_local_triangle(tri_vertices[j], gauss_number)
            gauss_weight = gauss.gauss_coefficient_local_triangle(tri_vertices[j], gauss_number)
            for v in np.arange(gauss_weight.shape[0]):  # 高斯积分，梯度的内积，就是x,y偏导乘积的和做积分
                val += (gauss_weight[v] * (
                        function.B_function(mk, self.centroid[i], self.hD[i], gauss_point[v], MI)
                        * pde.source(gauss_point[v])))
        return val
    def grad_mk_mk(self,mk,nk,i,MI,vertices,gauss_point_number):#梯度内积mk和nk的梯度内积
        function = Basis_function()
        # 获取分块三角形的顶点：
        tri_vertices = self.to_triangle(i, vertices)  # 第i个单元的分块三角形顶点
        tri_num = tri_vertices.shape[0]  # 三角形个数
        val = 0
        gauss = Triangle_Gauss_reference_2D()
        for j in np.arange(tri_num):
            # 获取积分点和积分系数
            gauss_point = gauss.gauss_point_local_triangle(tri_vertices[j], gauss_point_number)
            gauss_weight = gauss.gauss_coefficient_local_triangle(tri_vertices[j], gauss_point_number)
            for v in np.arange(gauss_weight.shape[0]):#高斯积分，梯度的内积，就是x,y偏导乘积的和做积分
                val += (gauss_weight[v] * (function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point[v], MI,0)
                        *function.B_function_derivative(nk, self.centroid[i], self.hD[i], gauss_point[v], MI,0)
                                           +function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point[v], MI,1)
                        *function.B_function_derivative(nk, self.centroid[i], self.hD[i], gauss_point[v], MI,1)) )
        return val
    def dof_mk(self,mk,dof,i,MI,vertices,gauss_point_number):#mk是多项式编号，dof是自由度编号,i是单元编号,MI存了m函数所有自由度次数多重指标
        """
        D_i\alpha中的元素(dof_N(m_{nk}))即m多项式在自由度处的取值，内部自由度是根据积分定义的
        """
        function = Basis_function()
        if dof <= self.cell[i].shape[0]-1:#如果自由度在边界自由度内
            function_m=function.B_function(mk,self.centroid[i],self.hD[i],self.node[self.cell[i,dof]],MI)
            return function_m
        else:#内部自由度需要积分，mk编号函数，和dof自由度导出的多项式乘积积分
            mfunction_num=dof-self.cell[i].shape[0]#m函数的多重指标编号
            #获取分块三角形的顶点：
            tri_vertices=self.to_triangle(i,vertices)#第i个单元的分块三角形顶点
            tri_num=tri_vertices.shape[0]#三角形个数
            val=0
            gauss = Triangle_Gauss_reference_2D()
            for j in np.arange(tri_num):
                #获取积分点和积分系数
                gauss_point=gauss.gauss_point_local_triangle(tri_vertices[j],gauss_point_number)
                gauss_weight=gauss.gauss_coefficient_local_triangle(tri_vertices[j],gauss_point_number)
                for v in np.arange(gauss_weight.shape[0]):
                    val+=(gauss_weight[v]*function.B_function(mfunction_num,self.centroid[i],self.hD[i],gauss_point[v],MI)
                          *function.B_function(mk,self.centroid[i],self.hD[i],gauss_point[v],MI))
            return val/np.abs(self.area[i])
    def grad_integration_m_phi(self,mk,dof,i,MI,gauss_point_number):#一维积分

        function = Basis_function()
        gauss=Gauss_Lobatto_reference_1D()
        mesh=quadrangle_mesh_2D()
        if dof < 4:
            if dof < 3 and dof >0:#仅支持四边形
                dof0=self.node[self.cell[i, dof-1]]
                dof1=self.node[self.cell[i,dof]]
                dof2=self.node[self.cell[i,dof+1]]
            elif dof == 0:
                dof0 = self.node[self.cell[i, 3]]
                dof1 = self.node[self.cell[i, dof]]
                dof2 = self.node[self.cell[i, dof + 1]]
            elif dof ==3:
                dof0 = self.node[self.cell[i, dof-1]]
                dof1 = self.node[self.cell[i, dof]]
                dof2 = self.node[self.cell[i, 0]]
            #其中一条边的积分
            endpoints_1=np.array([ dof1 , dof2 ] )
            gauss_point_1,gauss_weight_1=gauss.map_1D_to_2D_segment(endpoints_1,gauss_point_number)
            norm_out_1=mesh.normal_out(dof1,dof2)
            result_1=0
            for q in np.arange(gauss_weight_1.shape[0]):#自由度在这条边的起始点
                result_1 +=((function.B_function_derivative(mk,self.centroid[i],self.hD[i],gauss_point_1[q],MI,0)* norm_out_1[0]
                             +function.B_function_derivative(mk,self.centroid[i],self.hD[i],gauss_point_1[q],MI,1)* norm_out_1[1])
                             *gauss_weight_1[q]*self.delta_function(0,q))
            #第二条边的积分
            endpoints_2 = np.array([dof0,dof1])
            gauss_point_2, gauss_weight_2 = gauss.map_1D_to_2D_segment(endpoints_2, gauss_point_number)
            norm_out_2 = mesh.normal_out(dof0,dof1)
            result_2 = 0
            for q in np.arange(gauss_weight_2.shape[0]):  # 自由度在这条边的起始点
                result_2 += ((function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point_2[q], MI,
                                                               0) * norm_out_2[0]
                                + function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point_2[q], MI,
                                                                 1) * norm_out_2[1])
                               * gauss_weight_2[q] * self.delta_function(gauss_weight_2.shape[0]-1, q))
            return result_1+result_2

        elif dof>=4 and dof <self.cell[i].shape[0]:
            dof0=self.cell[i,dof-4]
            if dof < self.cell[i].shape[0]-1:
                dof1=self.cell[i,dof-3]
            elif dof ==self.cell[i].shape[0]-1:
                dof1=self.cell[i,0]
            endpoints = np.array([self.node[dof0],self.node[dof1]])

            gauss_point, gauss_weight = gauss.map_1D_to_2D_segment(endpoints, 3)
            norm_out_1 = mesh.normal_out(self.node[dof0], self.node[dof1] )
            result = 0
            for q in np.arange(gauss_weight.shape[0]):  # 自由度在这条边的起始点
                result += ((function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point[q], MI,
                                                             0) * norm_out_1[0]
                              + function.B_function_derivative(mk, self.centroid[i], self.hD[i], gauss_point[q], MI,
                                                               1) * norm_out_1[1])
                             * gauss_weight[q] * self.delta_function(1, q))
            return result

        else:#内部自由度，拉普拉斯mk 乘以phi，通过MI的次数，判断拉普拉斯是不是0，
            phi_nk=dof-self.cell[i].shape[0]
            if MI[mk,0]-MI[phi_nk,0]==2 :#仅针对另外一个变量拉普拉斯为0的情况，维度高了需要修改
                return -self.area[i]*((MI[mk,0]*(MI[mk,0]-1))/self.hD[i]**2 )
            elif MI[mk,1]-MI[phi_nk,1]==2:
                return -self.area[i]*((MI[mk,1]*(MI[mk,1]-1))/self.hD[i]**2 )
            else:return 0
    def integration_m_phi(self,mk,dof,i):
        if dof<self.cell[i].shape[0]:
            return 0
        elif mk==dof-self.cell[i].shape[0]:
            return self.area[i]
        else:return 0
    def matrix_C(self,i,k,H,TI,vertices):
        # 自由度
        count = Count_number()
        dof_ex = vertices[i].shape[0] + vertices[i].shape[0] * (k - 1)  # 边界自由度
        if k < 2:
            dof_in = 0
        else:
            dof_in = count.dim_p_k(k - 2, 2)  # 内部自由度，n_{k-2}
        nk = count.dim_p_k(k, 2)  # 多项式的总数
        # 总自由度
        dof = dof_in + dof_ex
        C_matrix=np.zeros((nk,dof))
        for j in np.arange(nk):
            if j <dof_in:#如果mk小于nk-2，利用自由度直接算
                for q in np.arange(dof):
                    C_matrix[j,q]=self.integration_m_phi(j,q,i)
            else:
                Mid=H @ TI
                Mid[:dof_in,:]=0
                C=C_matrix+Mid
                return C
    def matrix_B(self,i,k,MI,vertices,gauss_point_number):#实际上这个高斯积分点是固定的，因为边界上我们取了lobatto积分点作为自由度
        # 自由度
        count = Count_number()
        dof_ex = vertices[i].shape[0] + vertices[i].shape[0] * (k - 1)  # 边界自由度
        if k < 2:
            dof_in = 0
        else:
            dof_in = count.dim_p_k(k - 2, 2)  # 内部自由度，n_{k-2}
        nk = count.dim_p_k(k, 2)  # 多项式的总数
        # 总自由度
        dof = dof_in + dof_ex
        B_matrix=np.zeros((nk,dof))
        for j in np.arange(nk):
            if j ==0 :
                for p in np.arange(dof):
                    B_matrix[j, p] = self.operator_p0_phi(vertices,k,i,p)
            else:
                for p in np.arange(dof):
                    B_matrix[j,p]=self.grad_integration_m_phi(j,p,i,MI,gauss_point_number)
        return B_matrix
    def matrix_D(self,i,k,MI,vertices,gauss_point_number):#局部矩阵D
        #自由度
        count = Count_number()
        dof_ex=vertices[i].shape[0]+vertices[i].shape[0]*(k-1)#边界自由度
        if k <2:
            dof_in= 0
        else:
            dof_in=count.dim_p_k(k-2,2)#内部自由度，n_{k-2}
        nk=count.dim_p_k(k,2)#多项式的总数
        #总自由度
        dof=dof_in+dof_ex
        D_matrix=np.zeros((dof,nk))
        for j in np.arange(dof):
            for p in np.arange(nk):
                D_matrix[j,p]=self.dof_mk(p,j,i,MI,vertices,gauss_point_number)
        return D_matrix
    def matrix_G(self,i,k,MI,vertices,gauss_point_number,nk):#局部矩阵G,nk是多项式空间自由度
        G_matrix=np.zeros((nk,nk))
        for j in np.arange(nk):
            if j ==0 :
                for p in np.arange(nk):
                    G_matrix[j,p]=self.operator_p0_mk(p,k,i,MI,vertices,gauss_point_number)
            else:
                for p in np.arange(1,nk):
                    G_matrix[j,p]=self.grad_mk_mk(j,p,i,MI,vertices,gauss_point_number)
        return G_matrix
    def matrix_H(self,i,k,MI,vertices,gauss_number,nk):
        H_matrix=np.zeros((nk,nk))
        for j in np.arange(nk):
            for p in np.arange(nk):
                H_matrix[j,p]=self.integration_m_m(j,p,i,MI,vertices,gauss_number)
        return H_matrix
    def vector_f_local(self,i,MI,vertices,gauss_number,nk,TI,c):#TI=G^{-1}@B
        F_m=np.zeros((nk,1))
        for j in np.arange(nk):
            F_m[j]=self.integration_f_m(j,i,MI,vertices,gauss_number,c)
        F_vector=TI.T @ F_m

        return F_vector
    def stiffness_matrix_local(self,i,k,MI,vertices,nk,gauss_number):
        # 第i个单元的矩阵
        D = self.matrix_D(i, k, MI, vertices, gauss_number)  # 区域积分点
        G = self.matrix_G(i, k, MI, vertices, gauss_number, nk)  # 区域积分点
        B = self.matrix_B(i, k, MI, vertices, k + 1)  # 这个积分点是lobatto积分点.

        TI = np.linalg.inv(G) @ B  # TI=G^{-1}@B
        Ti = D @ TI
        G_tilde = np.zeros((G.shape[0], G.shape[1]))
        G_tilde[:] = G[:]
        G_tilde[0, :] = 0
        E = np.identity(D.shape[0])
        K = TI.T @ G_tilde @ TI + (E - Ti).T @ (E - Ti)
        return K,TI,D,G,B#GBD，是计算基函数投影的
    def mass_matrix_local(self,i,k,MI,TI,D,vertices,gauss_number,nk):
        H = self.matrix_H(i, k, MI, vertices, gauss_number, nk)

        C = self.matrix_C(i, k, H, TI, vertices)
        E=np.identity(D.shape[0])
        HC=np.linalg.inv(H) @ C
        I_TI=E-D@HC
        M=C.T @ HC+self.area[i] * (I_TI.T @ I_TI)
        return M,HC

    def matrix_A_F(self,k,MI,vertices,gauss_number,nk,TI,K,M,c_pde):#刚度矩阵和质量矩阵这些矩阵形状不变的时候就不会变化.
        cell_num=self.cell.shape[0]#单元数
        dof_ex=self.node.shape[0]#边界自由度直接node的维度就行
        in_c=Count_number().dim_p_k(k - 2, 2)
        if k <2 :dof_in=0#k<2时没有内部自由度
        else:dof_in = cell_num*in_c
        dof = dof_in+dof_ex
        A_matrix=np.zeros((dof,dof))
        F_matrix=np.zeros((dof,1))
        for c in np.arange(cell_num):
            F=self.vector_f_local(c,MI,vertices,gauss_number,nk,TI,c)

            for i in np.arange(K.shape[0]):
                for j in np.arange(K.shape[1]):
                    if i<self.cell[c].shape[0] and j<self.cell[c].shape[0]:
                        A_matrix[self.cell[c,i],self.cell[c,j]] +=K[i,j]+c_pde*M[i,j]
                    elif i >= self.cell[c].shape[0] and j<self.cell[c].shape[0]:
                        num_A_in=dof_ex-1+c*in_c+i-self.cell[c].shape[0]+1
                        A_matrix[num_A_in,self.cell[c,j]] += K[i,j]+c_pde*M[i,j]
                    elif i<self.cell[c].shape[0] and j>=self.cell[c].shape[0]:
                        num_A_in = dof_ex - 1 + c * in_c + j - self.cell[c].shape[0] + 1
                        A_matrix[self.cell[c,i],num_A_in] += K[i,j]+c_pde*M[i,j]
                    else:
                        num_A_in_i=dof_ex-1+c*in_c+i-self.cell[c].shape[0]+1
                        num_A_in_j=dof_ex - 1 + c * in_c + j - self.cell[c].shape[0] + 1
                        A_matrix[num_A_in_i,num_A_in_j] += K[i,j]+c_pde*M[i,j]
            for p in np.arange(F.shape[0]):
                if p < self.cell[c].shape[0]:
                    F_matrix[self.cell[c,p]]+=F[p,0]
                else:
                    num_F_in_p=dof_ex-1+c*in_c+p-self.cell[c].shape[0]+1
                    F_matrix[num_F_in_p] += F[p,0]
        return A_matrix,F_matrix

