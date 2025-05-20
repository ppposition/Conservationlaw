import numpy as np
import scipy.optimize as opt

def exact_solution(x, t):
    """
    计算Burgers方程的精确解
    
    参数:
        x: 空间点(数组或单个值)
        t: 时间点
    
    返回:
        u: 对应每个x点的精确解值
    """
    # 如果x是单个值，转为数组方便处理
    if np.isscalar(x):
        x = np.array([x])
    
    # 初始条件定义
    def u0(x0):
        return np.sin(np.pi * x0 + np.pi)
    
    # 特征线方程: x - x0 - t*u0(x0) = 0
    # 对于每个x，我们需要求解这个方程找到对应的x0
    x0_ex = np.zeros_like(x, dtype=float)
    
    for i in range(len(x)):
        # 定义特征线方程
        def f_char(x0):
            return (x[i] - x0) - t * u0(x0)
        
        # 根据x的符号选择初始猜测值
        if x[i] > 0:
            initial_guess = 1.0
        else:
            initial_guess = -1.0
        
        try:
            # 使用scipy的fzero等价函数求解非线性方程
            x0_ex[i] = opt.fsolve(f_char, initial_guess)[0]
            
            # 检查求解结果是否在有效范围内
            if abs(x0_ex[i]) > 1:
                # 如果超出范围，尝试不同的初始猜测值
                x0_guesses = np.linspace(-1, 1, 21)  # 尝试多个初始点
                valid_solution = False
                
                for guess in x0_guesses:
                    try:
                        x0_candidate = opt.fsolve(f_char, guess)[0]
                        if abs(x0_candidate) <= 1 and abs(f_char(x0_candidate)) < 1e-10:
                            x0_ex[i] = x0_candidate
                            valid_solution = True
                            break
                    except:
                        continue
                
                if not valid_solution:
                    raise ValueError(f"计算范围超出初值 (x={x[i]}, t={t})")
            
        except Exception as e:
            print(f"求解特征线方程失败: x={x[i]}, t={t}, 错误: {str(e)}")
            x0_ex[i] = np.nan
    
    # 使用求得的x0计算精确解
    u = u0(x0_ex)
    
    return u

def exact_solution_burgers(x, t, u0_func=None):
    """
    求解一般Burgers方程的精确解
    
    参数:
        x: 空间点坐标数组
        t: 时间点
        u0_func: 初始条件函数，默认为sin(π*x+π)
    
    返回:
        u: 在点x和时间t的精确解
    """
    if u0_func is None:
        # 默认初始条件
        u0_func = lambda x0: np.sin(np.pi * x0 + np.pi)
    
    # 如果x是单个值，转为数组方便处理
    if np.isscalar(x):
        x = np.array([x])
    
    # 初始化结果数组
    u = np.zeros_like(x, dtype=float)
    
    # 对每个空间点求解特征线方程
    for i in range(len(x)):
        # 定义要求解的特征线方程: x - x0 - t*u0(x0) = 0
        def f_char(x0):
            return (x[i] - x0) - t * u0_func(x0)
        
        # 使用二分法求解方程，需要找到适当的搜索区间
        # 因为特征线可能会相交(形成激波)，我们需要仔细处理
        
        # 首先尝试在较大范围内寻找解
        x_min, x_max = -2.0, 2.0
        
        # 检查区间端点的函数值符号是否相反
        f_min, f_max = f_char(x_min), f_char(x_max)
        
        try:
            # 如果函数值符号相反，可以使用二分法
            if f_min * f_max <= 0:
                x0_sol = opt.bisect(f_char, x_min, x_max)
                u[i] = u0_func(x0_sol)
            else:
                # 如果函数值符号相同，尝试寻找使函数值为零的点
                x0_sol = opt.newton(f_char, 0.0, maxiter=100)
                
                # 验证解的有效性
                if abs(f_char(x0_sol)) < 1e-10:
                    u[i] = u0_func(x0_sol)
                else:
                    # 尝试多个初始猜测点
                    valid_solution = False
                    for guess in np.linspace(x_min, x_max, 20):
                        try:
                            x0_candidate = opt.newton(f_char, guess, maxiter=50)
                            if abs(f_char(x0_candidate)) < 1e-10:
                                u[i] = u0_func(x0_candidate)
                                valid_solution = True
                                break
                        except:
                            continue
                    
                    if not valid_solution:
                        # 可能是激波区域，选择满足熵条件的解
                        u[i] = np.nan  # 标记为未找到解，后续处理
        except:
            # 数值求解失败，可能是在激波区域
            u[i] = np.nan
    
    # 处理激波区域 (这里采用简单标记，实际应用中需要更复杂的处理)
    # 如果有更多关于激波区域的信息，可以进一步完善
    nan_mask = np.isnan(u)
    if np.any(nan_mask):
        print(f"警告: 在时间 t={t} 发现可能的激波区域")
    
    return u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试函数
    x = np.linspace(-1, 1, 1000)
    
    # 不同时间点的解
    times = [0, 0.2, 0.4, 0.6]
    
    plt.figure(figsize=(10, 6))
    
    for t in times:
        try:
            u = exact_solution(x, t)
            plt.plot(x, u, label=f't = {t}')
        except ValueError as e:
            print(f"t = {t}: {e}")
    
    plt.grid(True)
    plt.legend()
    plt.title('Burgers方程的精确解')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.savefig('burgers_exact_solution.png')
    plt.show() 