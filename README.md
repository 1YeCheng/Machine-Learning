# 机器学习课程项目集

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-model%20evaluation-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557C)
![LIBSVM](https://img.shields.io/badge/LIBSVM-3.31-00599C)
![Status](https://img.shields.io/badge/status-coursework%20portfolio-blue)

本仓库包含两个本科阶段完成的机器学习课程项目，内容围绕核心算法实现、可复现实验流程、定量评估与可视化结果展开，展示对经典机器学习方法的理解和实验组织能力。

这些项目面向研究生申请场景整理，希望让导师能够快速看到我在机器学习方向的基础训练：包括数学模型实现、实验设计意识，以及不只依赖单一准确率指标来比较模型的习惯。

## 项目概览

| 项目 | 主题 | 主要方法 | 输出内容 |
|---|---|---|---|
| [`lab1`](lab1/README.md) | 西瓜 3.0a 数据集上的线性模型实验 | 逻辑回归、二次特征扩展、线性判别分析、留一法、分层 K 折、自助采样袋外评估 | 指标表、预测结果、混淆矩阵、PR 曲线 |
| [`lab2`](lab2/README.md) | SVM/SVR 实验与模型对比 | LIBSVM 分类/回归、BP 神经网络、手写 C4.5 决策树 | 模型文件、预测表、评估指标、决策边界、对比图 |

实验报告草稿和 Word 文档已通过 `.gitignore` 排除；仓库保留代码、数据集和可复现实验产物。

## 技术亮点

- 从零实现二分类逻辑回归，包括牛顿法优化、数值稳定的 sigmoid/softplus、带部分主元的高斯消元，以及可选 L2 正则化。
- 从零实现二分类线性判别分析，包括类别先验、共享协方差估计、协方差正则化，以及基于判别函数的概率预测。
- 构建多种评估流程，包括训练集回代、留一法交叉验证、分层 5 折交叉验证和 bootstrap 袋外评估。
- 集成 LIBSVM 完成 SVM 分类与 SVR 回归，覆盖缩放、训练、预测、支持向量分析和决策边界可视化。
- 在 UCI 数据集上比较 SVM、BP 神经网络和手写 C4.5 风格增益率决策树。

## 代表性结果

| 实验 | 数据集 | 代表性结果 |
|---|---|---|
| 二次特征逻辑回归 | Watermelon 3.0a | 5 折交叉验证 accuracy 0.7059，F1 0.7059 |
| 线性判别分析 | Watermelon 3.0a | 5 折交叉验证 accuracy 0.6471，F1 0.6250 |
| SVM 分类 | Watermelon 3.0a | Linear/RBF SVM accuracy 0.7059，F1 0.6154 |
| UCI 分类对比 | Wine | Linear SVM accuracy 0.9630；C4.5 F1 0.9482 |
| UCI 分类对比 | Breast Cancer WDBC | BP accuracy 0.9825；Linear SVM accuracy 0.9649 |
| SVR 回归 | Watermelon 3.0a | RBF-SVR RMSE 0.112893，MAE 0.095463 |

其中小规模西瓜数据集主要用于透明地分析算法机制和可视化模型行为；UCI 实验则提供了更大规模基准数据集上的横向比较。

## 仓库结构

```text
.
+-- lab1/
|   +-- 3.3/      # 逻辑回归实验
|   +-- 3.5/      # 线性判别分析实验
+-- lab2/
|   +-- data/     # 西瓜数据集与 UCI 数据集
|   +-- src/      # 实验流水线与手写 C4.5 决策树
|   +-- results/  # 生成的指标、预测结果和图像
|   +-- libsvm-3.31/
+-- README.md
```

## 运行环境

项目使用 Python 3.10+ 和常见科学计算包。`lab2` 额外依赖仓库中随附的 Windows 版 LIBSVM 可执行文件。

```bash
pip install numpy matplotlib scikit-learn
```

从各项目目录运行实验：

```bash
cd lab2
python src/run_lab2_experiments.py
```

`lab1` 中的脚本按实验组织：

```bash
cd lab1/3.3
python src/logistic_regression_watermelon.py
python src/logistic_regression_quadratic_5fold.py

cd ../3.5
python src/linear_discriminant_analysis_watermelon.py
```
