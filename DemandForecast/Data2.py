import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 엑셀 파일 경로
file_path = '../data2.xlsx'


# 특정 시트 읽기
data = pd.read_excel(file_path, sheet_name='2023')
data_numeric = data.select_dtypes(include='number')
ramen_data = data[data['대분류'] == '면류.라면류']
# print("ramen_data", ramen_data)
# corr_matrix = data_numeric.corr()

# 데이터프레임 출력
# print(data)

# print(ramen_data.info())
# print(ramen_data.shape)
# print(ramen_data.describe())

# print(ramen_data.isnull().sum())

# 결측치 최빈값 처리
print(ramen_data['소분류'].mode()[0])
ramen_data['소분류'] = ramen_data['소분류'].fillna(ramen_data['소분류'].mode()[0])
print(ramen_data.isnull().sum())

print(ramen_data)

# 불필요한 변수 제거.. 뭘 제거해야 하지?

# plt.rc('font', family='AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
# plt.show()

# print(data.corr(numeric_only=True).style.background_gradient())
