import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
class DataInspection:
    def __init__(self, df):
        self.df = df

    def show_columns(self):
        """显示数据集中的所有列。"""
        print("\nFollowing are the variables in your dataset:")
        for idx, col in enumerate(self.df.columns, 1):
            print(f"{idx}. {col}")

    def generate_statistics(self):
        """生成并返回每列的统计信息。"""
        stats_list = []

        for col in self.df.columns:
            col_type = "Nominal" if self.df[col].dtype == 'object' else (
                "Ordinal" if self.df[col].nunique() <= 10 else "Ratio"
            )
            if pd.api.types.is_numeric_dtype(self.df[col]):
                mean = self.df[col].mean()
                median = self.df[col].median()
                mode = self.df[col].mode()[0]
                kurtosis = self.df[col].kurt()
                skewness = self.df[col].skew()
                stats_list.append([
                    col, col_type, f"{mean:.2f} / {median:.2f} / {mode}", 
                    f"{kurtosis:.2f}", f"{skewness:.2f}"
                ])
            else:
                mode = self.df[col].mode()[0]
                stats_list.append([col, col_type, mode, "NA", "NA"])

        return stats_list

    def plot_boxplot(self, cont_var, cat_var):
        """生成箱形图并返回保存路径。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_var, y=cont_var, data=self.df, ax=ax)
        ax.set_title(f'Boxplot of {cont_var} by {cat_var}')

        # 保存图像并返回路径
        filepath = f'./results/anova_boxplot_{cont_var}_by_{cat_var}.png'
        fig.savefig(filepath)
        plt.close(fig)  # 关闭图形以释放内存
        print(f"Saved ANOVA boxplot as '{filepath}'.")
        return filepath  # 返回图片路径


    def plot_distribution(self, column):
        """绘制指定列的分布图并返回图片的完整路径。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        self.df[column].hist(bins=10, edgecolor='black', ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(False)

        # 保存图像到 ./results/ 目录
        filename = f'distribution_{column}.png'
        filepath = os.path.join('./results/', filename)
        fig.savefig(filepath)  # 保存图像
        plt.close(fig)  # 关闭图像
        print(f"Saved distribution plot as '{filepath}'.")

        return filepath  # 返回图片的完整路径


