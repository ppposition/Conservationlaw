# 传统方法与PINN求解Burgers方程

## 项目简介

本项目使用传统方法与PINN来求解具有激波和稀疏波的Burgers方程。

## 方法原理

### Burgers方程

Burgers方程是一个基本的非线性双曲型偏微分方程：

```
∂u/∂t + ∂(u²/2)/∂x = 0
```

该方程在初始条件存在不连续或梯度大的区域时会形成激波（shock wave）或稀疏波（rarefaction wave）。

### 传统方法
包括了 Lax-Friedrichs方法，Lax-Wendroff方法，ROE方法, Godunov方法。

### PINN方法

包括了标准PINN，XPINN， Shock-PINN, fPINN。

## 代码结构

- `xpinn.py`: XPINN模型实现及求解Burgers方程的主要代码
- `fpinn.py`: 基于傅里叶特征的PINN模型（用于比较）
- `shock_pinn.py`: Shock-PINN模型实现及求解Burgers方程的主要代码
- `conservation.ipynb`: 传统方法以及PINN，LBPINN

### 主要类

1. **`FourierFeatureLayer`**: 傅里叶特征变换层
2. **`SubNet`**: 子域神经网络
3. **`XPINN`**: 主XPINN模型，协调多个子网络
4. **`solve_burgers_with_xpinn`**: 求解Burgers方程的主函数

## 使用方法

### 安装依赖

```bash
pip install torch numpy matplotlib
```

### 运行示例

```bash
python xpinn.py
```

### 关键参数设置

- `n_domains`: 子域数量
- `scales`: 每个子域的傅里叶特征尺度
- `N_f`: PDE采样点总数
- `N_interface`: 接口采样点数量
- `epochs`: 训练轮数

## 示例问题

### 1. 激波问题

初始条件：
```
u(x,0) = { 1, x < 0
          { 0, x ≥ 0
```

解析解：
```
u(x,t) = { 1, x < t/2
          { 0, x ≥ t/2
```

### 2. 稀疏波问题

初始条件：
```
u(x,0) = { 0, x < 0
          { 1, x ≥ 0
```

解析解：
```
u(x,t) = { 0,   x < 0
          { x/t, 0 ≤ x < t
          { 1,   x ≥ t
```