import numpy as np
from scipy.special import comb
class Count_number:
    def __init__(self):
        pass
    #主要生成一些程序中必要参数，例如自由度个数，多项式维数等
    def dim_p_k(self,k,d):
        #多项式维数
        return int(comb((k+d),d))
    def multi_index_1(self, j):  # 通过索引获取次数，j从1开始
        T = 0
        deg = 0
        # 迭代直到T >= j
        while T < j:
            T = T + deg + 1
            deg += 1
            # 调整T和deg的值
        T = T - deg
        deg -= 1
        # 计算m和n
        m = deg - (j - T - 1)
        n = deg - m
        return m, n
    def multi_index_2(self,k,d):#向量版本，输入次数，得到多重指标
        dim=self.dim_p_k(k,d)
        M=np.zeros((dim,2))
        for i in np.arange(dim):
            M[i,0],M[i,1]=self.multi_index_1(i+1)
        return M






