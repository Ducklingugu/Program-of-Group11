import pandas as pd
from Module_Data_Inspection.DataInspection import DataInspection
from Module_data_analysis.DataAnalysis import DataAnalysis
from Module_SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 创建 results 目录（如果不存在）
os.makedirs('./results/', exist_ok=True)

def load_dataset():
    """加载数据集，并提示用户输入路径。"""
    path = input("ENTER THE PATH TO YOUR DATASET: ").strip('\"')
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return None

def show_image(image_path):
    """显示图片，如果路径存在则显示，否则提示用户文件不存在。"""
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    else:
        print(f"Image '{image_path}' does not exist.")

def print_statistics(stats_list):
    """打印数据统计信息。"""
    headers = ["Variable", "Type", "Mean / Median / Mode", "Kurtosis", "Skewness"]
    col_widths = [20, 10, 25, 10, 10]

    header_row = "".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
    print(header_row)
    print("=" * sum(col_widths))

    for stat in stats_list:
        print("".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(stat)))

def main_menu():
    """显示主菜单，并返回用户的选择。"""
    print("\nHow do you want to analyze your data?")
    print("1. Plot variable distribution")
    print("2. Conduct ANOVA")
    print("3. Conduct t-Test")
    print("4. Conduct Chi-Square")
    print("5. Conduct Regression")
    print("6. Conduct Sentiment Analysis")
    print("7. Quit")
    return input("Enter your choice (1 – 7): ")

def main():
    try:
        df = load_dataset()
        if df is None:
            return

        # 初始化模块
        inspection = DataInspection(df)
        analysis = DataAnalysis(df)
        sentiment = SentimentAnalysis(df)

        # 打印统计信息
        stats_list = inspection.generate_statistics()
        print_statistics(stats_list)

        while True:
            choice = main_menu()

            if choice == '1':  # 绘制变量分布
                try:
                    inspection.show_columns()
                    col_index = int(input("Enter the column number for distribution plot: ")) - 1
                    if col_index not in range(len(df.columns)):
                        raise ValueError("Invalid column number.")

                    column = df.columns[col_index]
                    image_path = inspection.plot_distribution(column)
                    show_image(image_path)

                except ValueError as e:
                    print(f"Error: {e}")
            elif choice == '2':  # ANOVA 或 Kruskal-Wallis
                try:
                    print("\nAvailable variables for ANOVA:")
                    for stat in stats_list:
                        if stat[1] in ['Ratio', 'Interval', 'Ordinal', 'Nominal']:
                            print(f"{stat[0]}\t{stat[1]}")

                    # 输入变量
                    cont_var = input("Enter a continuous (interval/ratio) variable:")
                    cat_var = input("Enter a categorical (ordinal/nominal) variable:")

                    # 检查变量是否存在于数据集中
                    if cont_var not in df.columns or cat_var not in df.columns:
                        raise ValueError("Invalid variable name(s).")

                    # 分类变量必须至少有 2 个不同组
                    if df[cat_var].nunique() < 2:
                        raise ValueError(f"'{cat_var}' must have at least 2 unique groups.")

                    # 检查连续变量是否为数值型
                    if not pd.api.types.is_numeric_dtype(df[cont_var]):
                        raise ValueError(f"'{cont_var}' must contain numeric values.")

                    # **正态性检验**
                    if not analysis.check_normality(df[cont_var], cont_var):
                        print(f"'{cont_var}' is not normally distributed.")
                        print("Performing Kruskal-Wallis Test instead...")
                        stat, p_value = analysis.perform_kruskal_wallis(cont_var, cat_var)
                    else:
                        print(f"'{cont_var}' normally distributed.")
                        print("Performing ANOVA.")
                        f_stat, p_value = analysis.anova(cont_var, cat_var)
                        print(f"ANOVA : F-statistic={f_stat}, p-value={p_value}")

                except ValueError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Unexpected error during ANOVA: {e}")

            elif choice == '3':  # t-Test
                try:
                    print("\nAvailable variables for t-Test:")
                    for stat in stats_list:
                        if stat[1] in ['Ratio', 'Interval']:
                            print(f"{stat[0]}\t{stat[1]}")

                    # 用户选择变量
                    num_var = input("Enter the numeric variable: ")
                    cat_var = input("Enter the categorical variable: ")

                    # 检查变量是否存在于数据集中
                    if num_var not in df.columns or cat_var not in df.columns:
                        raise ValueError("Invalid variable name(s). Please try again.")

                    # 执行 t-Test 或 Mann-Whitney U 测试
                    stat, p_value = analysis.t_test_or_mannwhitney(num_var, cat_var)
                    print(f"\nt-Test/Mann-Whitney U Test Results: stat={stat:.4f}, p-value={p_value:.4f}")

                    # 生成并显示箱形图
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x=cat_var, y=num_var, data=df)
                    plt.title(f'T-Test: {num_var} by {cat_var}')
                    plt.tight_layout()

                    # 保存并显示图像
                    filepath = f"./results/ttest_boxplot_{num_var}_by_{cat_var}.png"
                    plt.savefig(filepath)
                    plt.close()
                    show_image(filepath)  # 显示图像

                except ValueError as e:
                    print(f"Error during t-Test: {e}")

                except Exception as e:
                    print(f"Unexpected error during t-Test: {e}")

            elif choice == '4':  # Chi-Square
                try:
                    print("\nAvailable categorical variables:")
                    for stat in stats_list:
                        if stat[1] in ['Nominal', 'Ordinal']:
                            print(f"{stat[0]}\t{stat[1]}")

                    cat_var1 = input("Enter the first categorical variable: ")
                    cat_var2 = input("Enter the second categorical variable: ")

                    if cat_var1 not in df.columns or cat_var2 not in df.columns:
                        raise ValueError("Invalid variable name(s).")

                    stat, p_value = analysis.chi_square_test(cat_var1, cat_var2)
                    print(f"\nChi-Square Test Results: Statistic={stat}, p-value={p_value}")

                    # 生成并显示条形图
                    contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
                    plt.figure(figsize=(8, 6))
                    contingency_table.plot(kind="bar", stacked=True, colormap="viridis", ax=plt.gca())
                    plt.title(f"Chi-Square Test: {cat_var1} vs {cat_var2}")
                    plt.tight_layout()

                    # 保存并显示图像
                    filepath = f"./results/chi_square_{cat_var1}_vs_{cat_var2}.png"
                    plt.savefig(filepath)
                    plt.close()
                    show_image(filepath)  # 显示图像

                except ValueError as e:
                    print(f"Error: {e}")

                except Exception as e:
                    print(f"Unexpected error during Chi-Square Test: {e}")

            elif choice == '5':  # 回归分析
                try:
                    print("\nAvailable variables for Regression:")
                    for stat in stats_list:
                        if stat[1] in ['Ratio', 'Interval']:
                            print(f"{stat[0]}\t{stat[1]}")

                    dep_var = input("Enter the dependent variable: ")
                    ind_var = input("Enter the independent variable: ")

                    if dep_var not in df.columns or ind_var not in df.columns:
                        raise ValueError("Invalid variable name(s).")

                    analysis.regression(dep_var, ind_var)
                    show_image(f'./results/regression_{dep_var}_by_{ind_var}.png')

                except ValueError as e:
                    print(f"Error: {e}")

            elif choice == '6':  # 使用情感分析模块进行分析
                try:
                    # 获取并展示文本列
                    columns = sentiment.get_text_columns()
                    if columns.empty:
                        print("No suitable text columns available for sentiment analysis.")
                        continue

                    column_names = columns['Column Name'].tolist()
                    print(f"Available columns: {', '.join(column_names)}")
                    column_name = input("Select a column: ")

                    if column_name not in df.columns:
                        print("Invalid column name selected.")
                        continue

                    # 选择情感分析方法
                    analyzer_choice = input("Select analyzer: 1 (VADER), 2 (TextBlob), 3 (DistilBERT): ")

                    data = df[column_name].astype(str).fillna('')  # 确保数据是字符串格式

                    if analyzer_choice == '1':
                        scores, sentiments = sentiment.vader_sentiment_analysis(data)
                        print(f"VADER Sentiment Analysis Complete.\nSample Results: {sentiments[:10]}")

                    elif analyzer_choice == '2':
                        scores, sentiments, subjectivities = sentiment.textblob_sentiment_analysis(data)
                        print(f"TextBlob Sentiment Analysis Complete.\nSample Results: {sentiments[:10]}")

                    elif analyzer_choice == '3':
                        try:
                            scores, sentiments = sentiment.distilbert_sentiment_analysis(data)
                            print(f"DistilBERT Sentiment Analysis Complete.\nSample Results: {sentiments[:10]}")
                        except ImportError as e:
                            print(f"Error: {e}. Please ensure the transformers library is installed.")
                    else:
                        print("Invalid choice. Please try again.")

                except ValueError as e:
                    print(f"Error: {e}. Please ensure the column contains appropriate text data.")

                except Exception as e:
                    print(f"Unexpected error during sentiment analysis: {e}")

            elif choice == '7':  # 退出
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
   main()
