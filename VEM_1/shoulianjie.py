import numpy as np
import matplotlib.pyplot as plt


def showrate(h, Err, opt1='r-*', opt2='k.-', strErr='||u-u_h||'):
    """
    绘制误差曲线和收敛阶曲线的函数
    h: 网格尺寸列表
    Err: 对应网格尺寸的误差列表
    opt1: 误差曲线的线条属性（默认为'r-*'）
    opt2: 收敛阶曲线的线条属性（默认为'k.-'）
    strErr: 误差的标注字符串（默认为'||u-u_h||'）
    """
    # 防止误差为0的情况，因为log(0)会导致问题
    Err = np.where(Err == 0, 1e-16, Err)
    # 使用numpy的polyfit进行一次多项式拟合，得到拟合直线的斜率（即收敛阶）
    p = np.polyfit(np.log(h), np.log(Err), 1)
    r = p[0]
    s = 0.75 * Err[0] / h[0] ** r
    # 绘制误差曲线
    plt.loglog([1 / item for item in h], Err, opt1, linewidth=2)
    # 绘制收敛阶曲线
    plt.loglog([1 / item for item in h], s * h ** r, opt2, linewidth=1)
    plt.xlabel('log(1/h)')
    # 添加图例
    plt.legend([strErr, f'O (h^{{{r:.2f}}})'], loc='best')
    plt.show()

h_example = [0.1, 0.05, 0.025, 0.0125]  # 示例的网格尺寸列表
Err_example = [2.389e-4,2.992992e-5,3.743323e-6,4.679803e-7]  # 示例的误差列表
showrate(h_example, Err_example)
p12=np.log(Err_example[0]/Err_example[1])/np.log(h_example[0]/h_example[1])
p23=np.log(Err_example[1]/Err_example[2])/np.log(h_example[1]/h_example[2])
p34=np.log(Err_example[2]/Err_example[3])/np.log(h_example[2]/h_example[3])
print(p12,p23,p34)