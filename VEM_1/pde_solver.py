import numpy as np

from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from VEM_1.L2_error import VEM_error
from VEM_1.Matrix import Matrix
from VEM_1.Poisson_date import poisson_function
from VEM_1.count_number import Count_number
from VEM_1.quadrangle_mesh import quadrangle_mesh_2D
c=0
#获取pde信息
pde=poisson_function(c)#需要输入u的系数c
k=2
d=2
nx=20
ny=20
domain=pde.domain()
gauss_number=9
count=Count_number()
MI=count.multi_index_2(k,d)
nk=count.dim_p_k(k,d)

#获取矩形网格
mesh=quadrangle_mesh_2D(domain)
node,cell=mesh.rectangle(nx,ny,2)#输入nx，ny，以及内部等分数n
vertices=mesh.rectangle_vertices(node,cell)#一个三维数组，存了所有单元的四个顶点的坐标

boundary_x,boundary_y=mesh.boundary_retangle(nx,ny,1)
boundary_x=boundary_x.astype(int)
boundary_y=boundary_y.astype(int)
#需要每个单元的质心，直径，面积
centroid,hD,area=mesh.elements_top(vertices)#每个拓扑信息都是存了所有单元的多维数组
matrix=Matrix(domain,node,cell,centroid,hD,area)
#刚度矩阵
K_E,TI,D,G,B=matrix.stiffness_matrix_local(0,k,MI,vertices,nk,gauss_number)
#质量矩阵
M,HC=matrix.mass_matrix_local(0,k,MI,TI,D,vertices,gauss_number,nk)
#全局矩阵A，和载荷向量F
A,F=matrix.matrix_A_F(k,MI,vertices,gauss_number,nk,TI,K_E,M,c)
#边界处理
for j in np.arange(boundary_x.shape[0]):
    for i in boundary_x[j]:
        A[i, np.arange(A.shape[1]) != i] = 0
        A[i, i] = 1
        F[i] = pde.Dirichlet(node[i])
for q in np.arange(boundary_y.shape[0]):
    for p in boundary_y[q]:
        A[p, np.arange(A.shape[1]) != p] = 0
        A[p, p] = 1
        F[p] = pde.Dirichlet(node[p])
A_sparse = csr_matrix(A)#将普通矩阵转换为稀疏矩阵

uh=spsolve(A_sparse,F)#这里要求A的形式是稀疏矩阵的形式

error=VEM_error(centroid,hD,area,MI,cell,node)
L_error=error.L_wuqiong(uh,TI,nk,vertices,gauss_number,c,domain,k)
L2_error=error.L2_error(uh,TI,nk,vertices,gauss_number,c,domain,k)
print(L_error)
print(L2_error)
#画图
uh_u=uh[:node.shape[0]]
pde.plot_uh_vertex_only(node,cell,uh_u)

# 显示图形
x2 = np.linspace(0, 1, 100)
y2 = np.linspace(0, 1, 100)
X2, Y2 = np.meshgrid(x2, y2)

u=np.sin(np.pi*X2)*np.sin(np.pi * Y2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X2, Y2, u, cmap='rainbow')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z (Solution Value)')
plt.show()