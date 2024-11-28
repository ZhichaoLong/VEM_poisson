import numpy as np

from VEM_1.Basis_function import Basis_function
from VEM_1.count_number import Count_number
from VEM_1.gauss_lobatto_1D import Gauss_Lobatto_reference_1D
from VEM_1.quadrangle_mesh import quadrangle_mesh_2D
import scipy.integrate as spi


def solution(x,y):

    u_func = np.sin(np.pi * x) * np.sin(np.pi * y)
    return u_func
result, error = spi.dblquad(solution, 0, 1, 0, 1)
print(result,error)
"""
B = (1/12 * np.array([[0, 0, 0, 0,0,0,0,0,12],
                 [ -np.sqrt(2), np.sqrt(2), np.sqrt(2), -np.sqrt(2),0,4*np.sqrt(2),0,-4*np.sqrt(2),0],
                 [-np.sqrt(2), -np.sqrt(2), np.sqrt(2), np.sqrt(2),-4*np.sqrt(2),0,4*np.sqrt(2),0,0],
                 [1,1,1,1,0,4,0,4,-12],
                 [1,-1,1,-1,0,0,0,0,0],
                 [1,1,1,1,4,0,4,0,-12]]))
D = ( 1/24 * np.array([[24, -6*np.sqrt(2), -6*np.sqrt(2),3,3,3],
                      [24, np.sqrt(2),-np.sqrt(2) ],
                      [24, np.sqrt(2) ,np.sqrt(2) ],
                      [24, -np.sqrt(2), np.sqrt(2)]]) )
G= 0.5 * np.array([[2,0,0],[0,1,0],[0,0,1]])
G_1=0.5 * np.array([[0,0,0],[0,1,0],[0,0,1]])
E= np.identity(4)
K=(np.linalg.inv(G)@B).T @ G_1 @ (np.linalg.inv(G)@B)+(E-D@np.linalg.inv(G)@B).T @ (E-D@np.linalg.inv(G)@B)
print(K)

"""
"""测试函数"""
count=Count_number()
n_k=count.dim_p_k(2,2)
MI=count.multi_index_2(3,2)

pk_function=Basis_function()
xD=np.array([0.5,0.5])
x=np.array([1,2])
re=pk_function.B_function(4,xD,2,x,MI)

df=pk_function.B_function_derivative(5,xD,0.5,x,MI,0)
ddf=pk_function.B_function_second_derivative(4,xD,0.5,x,MI,1)
mesh=quadrangle_mesh_2D([0,1,0,1])
node,cell = mesh.rectangle(1,1,2)
#测试积分点和积分系数
duandian=np.array([[0.5,0.5],[0.5,1]])
gauss=Gauss_Lobatto_reference_1D()
point,xishu=gauss.map_1D_to_2D_segment(duandian, 3)











