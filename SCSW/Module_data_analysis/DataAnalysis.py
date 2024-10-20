import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

import seaborn as sns
import pandas as pd
import os

class DataAnalysis:
    def __init__(self, df):
        """初始化 DataAnalysis 类，传入数据集。"""
        self.df = df

    @staticmethod
    def check_normality(data, var_name):
        """检查数据的正态性，并生成 Q-Q 图。"""
        stat, p_value = stats.shapiro(data.dropna())  # 忽略缺失值

        # 生成并保存 Q-Q 图
        plt.figure()
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {var_name}")
        plt.tight_layout()

        # 保存图像
        filepath = f"./results/qq_plot_{var_name}.png"
        plt.savefig(filepath)
        plt.close()  # 关闭图像以释放内存
        print(f"Saved Q-Q plot as '{filepath}'.")

        print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}")
        return p_value > 0.05  # 返回 True 表示符合正态分布

    def perform_kruskal_wallis(self, cont_var, cat_var):
        """执行 Kruskal-Wallis 检验并生成箱形图。"""
        try:
            groups = [self.df[cont_var][self.df[cat_var] == group].dropna() for group in self.df[cat_var].unique()]
            stat, p_value = stats.kruskal(*groups)

            # 生成并保存箱形图
            plt.figure()
            sns.boxplot(x=cat_var, y=cont_var, data=self.df)
            plt.title(f"Kruskal-Wallis Test: {cont_var} by {cat_var}")
            plt.tight_layout()

            filepath = f"./results/kruskal_boxplot_{cont_var}_by_{cat_var}.png"
            plt.savefig(filepath)
            plt.close()
            print(f"Saved Kruskal-Wallis boxplot as '{filepath}'.")

            print(f"Kruskal-Wallis Result: Statistic={stat}, p-value={p_value}")
            return stat, p_value
        except Exception as e:
            print(f"Error during Kruskal-Wallis Test: {e}")
            return None, None


    def t_test_or_mannwhitney(self, num_var, cat_var):
        """执行 t-Test 或 Mann-Whitney U 检验。"""
        # 获取分类变量的唯一类别
        unique_groups = self.df[cat_var].dropna().unique()
        if len(unique_groups) != 2:
            raise ValueError(f"t-Test requires exactly 2 groups, but '{cat_var}' has {len(unique_groups)} groups.")

        # 根据分类变量将数据拆分为两组
        group1 = self.df[self.df[cat_var] == unique_groups[0]][num_var].dropna()
        group2 = self.df[self.df[cat_var] == unique_groups[1]][num_var].dropna()

        # 检查每组数据是否足够进行 t-Test
        if group1.empty or group2.empty:
            raise ValueError(f"One of the groups for '{cat_var}' is empty. Ensure both groups have data.")

        # 检查正态性
        normal1 = self.check_normality(group1, num_var)
        normal2 = self.check_normality(group2, num_var)

        # 根据正态性选择检验方法
        if normal1 and normal2:
            print("Both groups are normally distributed. Performing t-Test...")
            stat, p_value = stats.ttest_ind(group1, group2)
        else:
            print("Data not normally distributed. Performing Mann-Whitney U Test...")
            stat, p_value = stats.mannwhitneyu(group1, group2)

        # 打印结果
        print(f"T-Test/Mann-Whitney U Test Results: stat={stat:.4f}, p-value={p_value:.4f}")

        # 生成并保存箱形图
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cat_var, y=num_var, data=self.df)
        plt.title(f'T-Test: {num_var} by {cat_var}')
        plt.tight_layout()

        # 保存图像
        filepath = f"./results/ttest_boxplot_{num_var}_by_{cat_var}.png"
        plt.savefig(filepath)
        plt.close()
        print(f"Saved t-Test boxplot as '{filepath}'.")

        return stat, p_value

    def chi_square_test(self, cat_var1, cat_var2):
        """执行 Chi-Square 检验并生成条形图。"""
        try:
            contingency_table = pd.crosstab(self.df[cat_var1], self.df[cat_var2])
            stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            # 生成并保存条形图
            plt.figure(figsize=(8, 6))
            contingency_table.plot(kind="bar", stacked=True, colormap="viridis", ax=plt.gca())
            plt.title(f"Chi-Square Test: {cat_var1} vs {cat_var2}")
            plt.tight_layout()

            filepath = f"./results/chi_square_{cat_var1}_vs_{cat_var2}.png"
            plt.savefig(filepath)
            plt.close()
            print(f"Saved Chi-Square bar plot as '{filepath}'.")

            print(f"Chi-Square Result: Statistic={stat}, p-value={p_value}")
            return stat, p_value
        except Exception as e:
            print(f"Error during Chi-Square Test: {e}")
            return None, None

    def regression(self, dep_var, ind_var):
        """执行线性回归分析并生成散点图及回归线图。"""
        try:
            # 提取因变量和自变量
            y = self.df[dep_var]
            X = self.df[ind_var]

            # 添加常数项（截距）
            X = sm.add_constant(X)

            # 拟合线性回归模型
            model = sm.OLS(y, X).fit()

            # 打印回归结果
            print(model.summary())

            # 生成散点图和回归线，使用 Seaborn 的 regplot
            plt.figure(figsize=(10, 6))
            sns.regplot(x=ind_var, y=dep_var, data=self.df, scatter_kws={'alpha': 0.2}, line_kws={"color": "red"})
            plt.title(f'Regression Analysis: {dep_var} ~ {ind_var}')
            plt.xlabel(ind_var)
            plt.ylabel(dep_var)
            plt.tight_layout()

            # 保存图像
            filepath = f"./results/regression_{dep_var}_by_{ind_var}.png"
            plt.savefig(filepath)
            plt.close()  # 关闭图像以释放内存
            print(f"Saved regression plot as '{filepath}'.")

        except Exception as e:
            print(f"Error during regression analysis: {e}")

