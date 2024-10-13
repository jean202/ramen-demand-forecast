import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 엑셀 파일 경로
file_path = '../data2.xlsx'


# 특정 시트 읽기
data = pd.read_excel(file_path, sheet_name='2023')
data_numeric = data.select_dtypes(include='number')
corr_matrix = data_numeric.corr()

# 데이터프레임 출력
# print(data)

# print(data.info())
# print(data.shape)
# print(data.describe())

print(data.isnull().sum())

# plt.rc('font', family='AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
# plt.show()

# print(data.corr(numeric_only=True).style.background_gradient())
