import numpy as np

from VEM_1.count_number import Count_number


class Basis_function:
    def __init__(self):
        pass
    """
    生成多项式空间的基函数
    """
    def B_function(self,nk,xD,hD,x,MI):
        """
        nk是多重指标索引，代表第几个基函数
        xD是质点坐标，是个向量
        hD是单元直径
        x是坐标，应该输入向量
        """
        dim_d=xD.shape[0]
        m=1
        for i in np.arange(dim_d):
            """索引范围警告"""
            if nk>MI.shape[0]-1:
                w=MI.shape[0]-1
                print(f"警告：输入的多重指标索引nk({nk})超过了MI的索引范围({w})，请检查输入参数！")
                raise ValueError("输入的多重指标索引超出了MI的范围。")
            m=m*(((x[i]-xD[i])/hD)**MI[nk,i])
        return m
    def B_function_derivative(self, nk, xD, hD, x, MI, d_num):
        """
        nk是多重指标索引，代表第几个基函数
        xD是质点坐标，是个向量
        hD是单元直径
        x是坐标，应该输入向量
        d_num是要对哪个分量求偏导的参数，取值范围应该在0到xD.shape[0] - 1之间
        MI存了每个自变量对应的次数，是多重指标，行指标是基函数个数，列是对应基函数的每个分量的次数
        """
        dim_d = xD.shape[0]

        # 检查输入的d_num是否在有效范围内
        if d_num < 0 or d_num >= dim_d:
            raise ValueError("输入的d_num超出了有效范围，请检查输入参数！")

        if nk > MI.shape[0] - 1:
            w = MI.shape[0] - 1
            print(f"警告：输入的多重指标索引nk({nk})超过了MI的索引范围({w})，请检查输入参数！")
            raise ValueError("输入的多重指标索引超出了MI的范围。")

        m_derivative = 1
        for i in np.arange(dim_d):
            if i != d_num:
                m_derivative *= (((x[i] - xD[i])/ hD) ** MI[nk, i])
            else:
                if MI[nk, d_num] > 0:
                    m_derivative *= MI[nk, d_num]/hD * ((x[i] - xD[i])/ hD) ** (MI[nk, d_num] - 1)
                else:
                    m_derivative *= 0  # 对指定求偏导的分量，次数为0时导数为0

        return m_derivative
    def B_function_second_derivative(self, nk, xD, hD, x, MI, d_num):
        """
        二阶偏导
        nk是多重指标索引，代表第几个基函数
        xD是质点坐标，是个向量
        hD是单元直径
        x是坐标，应该输入向量
        d_num是要对哪个分量求二阶偏导的参数，取值范围应该在0到xD.shape[0] - 1之间
        MI存了每个自变量对应的次数，是多重指标，行指标是基函数个数，列是对应基函数的每个分量的次数
        """
        dim_d = xD.shape[0]

        # 检查输入的d_num是否在有效范围内
        if d_num < 0 or d_num >= dim_d:
            raise ValueError("输入的d_num超出了有效范围，请检查输入参数！")

        if nk > MI.shape[0] - 1:
            w = MI.shape[0] - 1
            print(f"警告：输入的多重指标索引nk({nk})超过了MI的索引范围({w})，请检查输入参数！")
            raise ValueError("输入的多重指标索引超出了MI的范围。")

        m_second_derivative = 1

        for i in np.arange(dim_d):
            if i != d_num:
                if MI[nk, i] > 0:
                    m_second_derivative *= ((x[i] - xD[i]) / hD) ** MI[nk, i]
                else:
                    m_second_derivative *= 1
            else:
                if MI[nk, d_num] > 1:
                    m_second_derivative *= MI[nk, d_num] * (MI[nk, d_num] - 1) * ((x[d_num] - xD[d_num]) / hD) ** (
                                MI[nk, d_num] - 2) / (hD ** 2)

                else:
                    m_second_derivative *= 0

        return m_second_derivative