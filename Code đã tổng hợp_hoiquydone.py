import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import levene
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tabulate import tabulate
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.interpolate import splrep, splev


# =============================================================
# ĐỌC VÀ XỬ LÍ DỮ LIỆU
# =============================================================

print("PHẦN I. BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU\n")
print(" 1. Đọc dữ liệu từ file CSV: 'D:/data.csv'\n") 
#new_DF = pd.read_csv("D:/data.csv")
new_DF = pd.read_csv(r"D:/Bách khoa/242/code xstk/data.csv")
print("Dữ liệu mẫu:")
print(new_DF.head(), "\n")

print(" 2. Kiểm tra kiểu dữ liệu ban đầu:\n")
print(new_DF.dtypes, "\n")

print(" 3. Kiểm tra và xử lý dữ liệu khuyết:\n")
missing_count = new_DF.isnull().sum()
missing_cols = missing_count[missing_count > 0]

if not missing_cols.empty:
    print("Các cột có dữ liệu khuyết:")
    print(missing_cols, "\n")
    print("Số dòng có dữ liệu khuyết:", new_DF.isnull().any(axis=1).sum(), "\n")

    for col in missing_cols.index:
        if new_DF[col].dtype == 'object':
            fill_value = new_DF[col].mode()[0]
            new_DF[col].fillna(fill_value, inplace=True)
            print(f"→ Cột '{col}' (chuỗi): đã điền giá trị mode là '{fill_value}'.")
        else:
            fill_value = new_DF[col].mean()
            new_DF[col].fillna(fill_value, inplace=True)
            print(f"→ Cột '{col}' (số): đã điền giá trị trung bình là {fill_value:.2f}.")
else:
    print(" Không có dữ liệu khuyết.\n")

print()

if {'layer_height', 'wall_thickness'}.issubset(new_DF.columns):
    new_DF['height_x_thickness'] = new_DF['layer_height'] * new_DF['wall_thickness']
    print(" 4. Đã thêm biến tương tác 'height_x_thickness' = layer_height × wall_thickness.\n")
else:
    print(" 4. Không đủ cột để tạo biến tương tác 'height_x_thickness'.\n")

cols_to_drop = ['roughness', 'elongation', 'Unnamed: 0', 'id']
dropped = [col for col in cols_to_drop if col in new_DF.columns]
new_DF.drop(columns=dropped, inplace=True)
print(f" 5. Đã xóa {len(dropped)} cột không cần thiết: {', '.join(dropped)}\n")

selected_vars = [
    'fan_speed',
    'wall_thickness',
    'infill_density',
    'infill_pattern',
    'nozzle_temperature',
    'bed_temperature',
    'print_speed',
    'material',
    'layer_height',
    'height_x_thickness',
    'tension_strenght'
]

existing_vars = [v for v in selected_vars if v in new_DF.columns]
new_DF = new_DF[existing_vars].copy()
print(" 6. Tạo tập dữ liệu phân tích gồm các biến sau:")
for v in existing_vars:
    print(f" - {v}")
print()

print(" 7. Kiểu dữ liệu sau xử lý:\n")
print(new_DF.dtypes)

print("\n 8. Dữ liệu mẫu sau khi xử lý:\n")
print(new_DF.head())

# Tạo new_DF_encoded từ new_DF và mã hóa biến phân loại
new_DF_encoded = new_DF.copy()

# Label encode biến phân loại nếu chưa encode
if 'material' in new_DF.columns and new_DF['material'].dtype == 'object':
    le = LabelEncoder()
    new_DF_encoded['material'] = le.fit_transform(new_DF['material'])

# Nếu vẫn còn infill_pattern, encode luôn nếu cần
if 'infill_pattern' in new_DF.columns and new_DF['infill_pattern'].dtype == 'object':
    le2 = LabelEncoder()
    new_DF_encoded['infill_pattern'] = le2.fit_transform(new_DF['infill_pattern'])

# =============================================================
# PHÂN TÍCH THỐNG KÊ MÔ TẢ
# =============================================================

print("="*70)
print("PHẦN II. THỐNG KÊ MÔ TẢ")
print("="*70)
print("Trong phần này, chúng ta sẽ mô tả đặc điểm phân bố của các biến định lượng\ntrong bộ dữ liệu in 3D, bao gồm trung bình, độ lệch chuẩn, tứ phân vị và khoảng giá trị.\n")

columns = [
    "wall_thickness", "infill_density",
    "nozzle_temperature", "bed_temperature", "print_speed",
    "fan_speed", "tension_strenght"
]

def new_function(x):
    return pd.Series({
        'n': len(x),
        'xtb': x.mean(),
        'sd': x.std(),
        'Q1.25%': x.quantile(0.25),
        'Q2': x.median(),
        'Q3.75%': x.quantile(0.75),
        'min': x.min(),
        'max': x.max()
    })

summary_table = new_DF[columns].apply(new_function).T
summary_table.reset_index(inplace=True)
summary_table.columns = ['Variable', 'n', 'xtb', 'sd', 'Q1.25%', 'Q2', 'Q3.75%', 'min', 'max']

print("\nBẢNG 1. Bảng thống kê mô tả cho các biến liên tục:\n")
print(summary_table)


print("\nBẢNG 2. Một số thống kê cụ thể cho 3 biến quan trọng: Wall Thickness, Infill Density và Tension Strength.\n")

# Wall Thickness
aveWall_Thickness = new_DF['wall_thickness'].mean()
medWall_Thickness = new_DF['wall_thickness'].median()
sdWall_Thickness = new_DF['wall_thickness'].std()
maxWall_Thickness = new_DF['wall_thickness'].max()
minWall_Thickness = new_DF['wall_thickness'].min()

# Infill Density
aveinfill_density = new_DF['infill_density'].mean()
medinfill_density = new_DF['infill_density'].median()
sdinfill_density = new_DF['infill_density'].std()
maxinfill_density = new_DF['infill_density'].max()
mininfill_density = new_DF['infill_density'].min()

# Tension Strength
avetension_strenght = new_DF['tension_strenght'].mean()
medtension_strenght = new_DF['tension_strenght'].median()
sdtension_strenght = new_DF['tension_strenght'].std()
maxtension_strenght = new_DF['tension_strenght'].max()
mintension_strenght = new_DF['tension_strenght'].min()

statistical_table1 = pd.DataFrame({
    "Average": [aveWall_Thickness, aveinfill_density, avetension_strenght],
    "Median": [medWall_Thickness, medinfill_density, medtension_strenght],
    "Standard_Deviation": [sdWall_Thickness, sdinfill_density, sdtension_strenght],
    "Max": [maxWall_Thickness, maxinfill_density, maxtension_strenght],
    "Min": [minWall_Thickness, mininfill_density, mintension_strenght]
}, index=["Wall_Thickness", "Infill_density", "tension_strenght"])

print(statistical_table1)


print("\nPHÂN TÍCH TẦN SUẤT:\nDưới đây là bảng tần suất cho các biến rời để khảo sát sự phổ biến của các giá trị.")

print("\nTần suất Nozzle Temperature:")
print(new_DF['nozzle_temperature'].value_counts())

print("\nTần suất Layer Height:")
print(new_DF['layer_height'].value_counts())

print("\nTần suất Print Speed:")
print(new_DF['print_speed'].value_counts())

print("\nTần suất Material:")
print(new_DF['material'].value_counts())

print("\nTần suất Fan Speed:")
print(new_DF['fan_speed'].value_counts())


print("\nTRỰC QUAN HÓA DỮ LIỆU")
print("\n1. Biểu đồ Histogram cho biến Tension Strength\n→ Giúp hiểu rõ hơn về phân bố của biến này.")

plt.figure()
new_DF['tension_strenght'].hist()
plt.title('Histogram of Tension Strength')
plt.xlabel('Tension Strength')
plt.ylabel('Frequency')
plt.show()

print("\n2. Biểu đồ Boxplot so sánh Tension Strength theo từng yếu tố:")
factors = ['wall_thickness', 'print_speed', 'fan_speed']
for factor in factors:
    print(f" - {factor.replace('_', ' ').title()}")
    plt.figure()
    sns.boxplot(x=new_DF[factor], y=new_DF['tension_strenght'])
    plt.title(f"Boxplot: Tension Strength vs {factor.replace('_', ' ').title()}")
    plt.xlabel(factor.replace('_', ' ').title())
    plt.ylabel('Tension Strength')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n3. Biểu đồ Tương quan (Scatter Matrix) giữa Tension Strength và các yếu tố liên quan.")
from pandas.plotting import scatter_matrix

print("→ Tension Strength vs Print Speed")
scatter_matrix(new_DF[['tension_strenght', 'print_speed']], figsize=(6, 6))
plt.show()

print("→ Tension Strength vs Wall Thickness")
scatter_matrix(new_DF[['tension_strenght', 'wall_thickness']], figsize=(6, 6))
plt.show()

print("→ Tension Strength vs Fan Speed")
scatter_matrix(new_DF[['tension_strenght', 'fan_speed']], figsize=(6, 6))
plt.show()


# =============================================================
# PHÂN TÍCH THỐNG KÊ SUY DIỄN
# =============================================================

print("="*70)
print("PHẦN III. THỐNG KÊ SUY DIỄN")
print("="*70)

# BÀI TOÁN MỘT MẪU: Xây dựng khoảng tin cậy
print("\nPHÂN TÍCH MỘT MẪU: XÂY DỰNG KHOẢNG TIN CẬY CHO TRUNG BÌNH")


print("\nBước 1: Tính toán thống kê mẫu")
n = len(new_DF['tension_strenght'])
xtb = new_DF['tension_strenght'].mean()
s = new_DF['tension_strenght'].std()
stats_new_DF = pd.DataFrame({'Số lượng mẫu (n)': [n], 'Trung bình mẫu (x̄)': [xtb], 'Độ lệch chuẩn (s)': [s]})
print(stats_new_DF.to_string(index=False))

# QQ Plot
print("\nBiểu đồ QQ plot để kiểm tra phân phối chuẩn của dữ liệu:")
plt.figure(figsize=(8, 6))
stats.probplot(new_DF['tension_strenght'], dist="norm", plot=plt)
plt.title('Normal QQ Plot of tension_strenght')
plt.grid(True)
plt.show()


print("\nBước 2: Kiểm định giả thiết phân phối chuẩn (Shapiro-Wilk Test)")
stat, p_value = stats.shapiro(new_DF['tension_strenght'])
print(f"  - W-statistic: {stat:.4f}")
print(f"  - p-value    : {p_value:.4f}")
if p_value < 0.05:
    print("  => Dữ liệu KHÔNG tuân theo phân phối chuẩn (bác bỏ H0)")
else:
    print("  => Dữ liệu có thể được xem là tuân theo phân phối chuẩn (không bác bỏ H0)")


print("\nBước 3: Tính toán ngưỡng sai số (Epsilon) và Khoảng tin cậy 95%")
epsilon = stats.t.ppf(1 - 0.05/2, df=n-1) * (s / np.sqrt(n))
print(f"  - Ngưỡng sai số epsilon: {epsilon:.4f}")
CI = pd.DataFrame({
    'Cận dưới (Lower bound)': [xtb - epsilon],
    'Trung bình mẫu (x̄)': [xtb],
    'Cận trên (Upper bound)': [xtb + epsilon]
})
print("\nKhoảng tin cậy 95% cho trung bình sức căng bề mặt:")
print(CI.to_string(index=False))


#  BÀI TOÁN HAI MẪU: So sánh hai nhóm

print("\n\nKIỂM ĐỊNH HAI MẪU ĐỘC LẬP: ABS vs PLA")

print("\nBước 1: Phát biểu giả thuyết")
print("  - H0: Không có sự khác biệt về sức căng bề mặt giữa ABS và PLA")
print("  - H1: Có sự khác biệt về sức căng bề mặt giữa ABS và PLA")


print("\nBước 2: Kiểm tra giả định phân phối chuẩn")
for material in ['abs', 'pla']:
    print(f"\n  - Vật liệu: {material.upper()}")
    data = new_DF[new_DF['material'] == material]
    
    plt.figure(figsize=(8, 6))
    stats.probplot(data['tension_strenght'], dist="norm", plot=plt)
    plt.title(f'Normal QQ Plot of {material.upper()} tension_strenght')
    plt.grid(True)
    plt.show()

    stat, p_value = stats.shapiro(data['tension_strenght'])
    print(f"    + W-statistic: {stat:.4f}")
    print(f"    + p-value    : {p_value:.4f}")
    if p_value < 0.05:
        print(f"    => Dữ liệu {material.upper()} KHÔNG tuân theo phân phối chuẩn (bác bỏ H0)")
    else:
        print(f"    => Dữ liệu {material.upper()} có thể được xem là tuân theo phân phối chuẩn")


print("\nBước 3: Kiểm định giả định phương sai")

abs_data = new_DF[new_DF['material'] == 'abs']  
pla_data = new_DF[new_DF['material'] == 'pla']  
s1 = np.std(abs_data['tension_strenght'], ddof=1)
s2 = np.std(pla_data['tension_strenght'], ddof=1)
ratio = s1 / s2
print("\n#1: Quy tắc tỉ lệ phương sai s1/s2 ∈ [0.5; 2]")
print(f"   - s1 (ABS): {s1:.4f}")
print(f"   - s2 (PLA): {s2:.4f}")
print(f"   - Tỉ số s1/s2: {ratio:.4f}")
if 0.5 <= ratio <= 2:
    print("   => Có thể giả định phương sai hai mẫu bằng nhau.")
else:
    print("   => Có thể giả định phương sai hai mẫu KHÁC nhau.")

print("\n#2: Levene Test (Kiểm định phương sai)")
f_statistic, p_value_f = stats.levene(abs_data['tension_strenght'], pla_data['tension_strenght'])
print(f"   - Levene statistic: {f_statistic:.4f}")
print(f"   - p-value         : {p_value_f:.4f}")
if p_value_f < 0.05:
    print("   => Phương sai KHÁC nhau (bác bỏ H0)")
else:
    print("   => Không có đủ bằng chứng để bác bỏ H0, phương sai có thể bằng nhau")

print("\nBước 4: Thực hiện kiểm định T-test hai mẫu độc lập")
t_statistic, p_value_t = stats.ttest_ind(abs_data['tension_strenght'], pla_data['tension_strenght'], equal_var=True)
print(f"   - T-statistic: {t_statistic:.4f}")
print(f"   - p-value    : {p_value_t:.4f}")
if p_value_t < 0.05:
    print("   => Có sự khác biệt có ý nghĩa thống kê giữa hai vật liệu (bác bỏ H0)")
else:
    print("   => Không có sự khác biệt có ý nghĩa thống kê (không bác bỏ H0)")

# =============================================================
# PHÂN TÍCH THỐNG KÊ MÔ SUY DIỄN
# =============================================================

print("="*70)
print("PHẦN IV. PHÂN TÍCH PHƯƠNG SAI 1 NHÂN TỐ")
print("="*70)

print("\nBước 1: Phát biểu giả thuyết")
print("  Gọi µ1, µ2, µ3 lần lượt là sức căng bề mặt trung bình của bản in ở nhóm 1, 2, 3.")
print("  - Giả thuyết H0: µ1 = µ2 = µ3")
print("  - Giả thuyết H1: Tồn tại ít nhất một cặp µi ≠ µj với 1 ≤ i, j ≤ 3")

print("\nBước 2: Xác định các giả định cần kiểm tra khi thực hiện phân tích phương sai")
print("  • Giả định 1: Sức căng bề mặt ở 3 nhóm có phân phối chuẩn.")
print("  • Giả định 2: Phương sai sức căng bề mặt ở 3 nhóm bằng nhau.")
print("  • Giả định 3: Các quan sát được lấy mẫu độc lập.")  

print("\nBước 3: Kiểm tra giả định 1 bằng đồ thị QQ-Plot và kiểm định shapiro.test")
print("  - Giả định 1: Sức căng bề mặt ở 3 nhóm có phân phối chuẩn")
new_DF['nozzle_temperature_2'] = new_DF['nozzle_temperature'].apply(
    lambda temp: "Group_1" if temp < 220 else ("Group_3" if temp > 230 else "Group_2")
)

print("\nKiểm tra phân phối chuẩn cho từng nhóm")
Group_1 = new_DF[new_DF['nozzle_temperature_2'] == "Group_1"]
Group_2 = new_DF[new_DF['nozzle_temperature_2'] == "Group_2"]
Group_3 = new_DF[new_DF['nozzle_temperature_2'] == "Group_3"]
def normality_test(data, group_name):
    print(f"Kiểm tra phân phối chuẩn cho {group_name}:")
    #QQ-plot
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Normal Q-Q Plot - {group_name}")
    plt.show()
    #Shapiro-Wilk test
    shapiro_test = stats.shapiro(data)
    print(f"Shapiro-Wilk test: statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

    if shapiro_test.pvalue > 0.05:
        print(f"Kết luận: Dữ liệu của nhóm {group_name} có vẻ tuân theo phân phối chuẩn (không bác bỏ H0)")
    else:
        print(f"Kết luận: Dữ liệu của nhóm {group_name} không tuân theo phân phối chuẩn (bác bỏ H0)")
    print("-" * 30)


# Kiểm tra cho từng nhóm
normality_test(Group_1['tension_strenght'], "Group_1")
normality_test(Group_2['tension_strenght'], "Group_2")
normality_test(Group_3['tension_strenght'], "Group_3")
print("  - Giả định 2: Phương sai sức căng bề mặt ở 3 nhóm bằng nhau")

print("\nBước 4: Kiểm tra giả định phương sai bằng nhau bằng kiểm định Levene")
levene_test = levene(Group_1['tension_strenght'], Group_2['tension_strenght'], Group_3['tension_strenght']) 
print(f"Levene's Test: statistic={levene_test.statistic}, p-value={levene_test.pvalue}")
if levene_test.pvalue > 0.05:
    print("Kết luận: Phương sai sức căng bề mặt ở 3 nhóm bằng nhau (không bác bỏ H0)")
else:
    print("Kết luận: Phương sai sức căng bề mặt ở 3 nhóm không bằng nhau (bác bỏ H0)")


#ANOVA model
print("\nBước 5: Thực hiện phân tích phương sai một nhân tố")
anova_model = ols('tension_strenght ~ C(nozzle_temperature_2)', data=new_DF).fit()

# Summary của ANOVA
anova_results = anova_lm(anova_model)
print(anova_results)
p_value_anova = anova_results['PR(>F)'].iloc[0] 
print(f"Giá trị p-value của ANOVA là: {p_value_anova:.4f}")
if p_value_anova < 0.05:  
    print("Kết luận: Có sự khác biệt có ý nghĩa thống kê về sức căng bề mặt giữa các nhóm nhiệt độ vòi phun.")
else:
    print("Kết luận: Không có sự khác biệt có ý nghĩa thống kê về sức căng bề mặt giữa các nhóm nhiệt độ vòi phun.")

print("\nBước 6: Thực hiện so sánh bội để làm rõ hơn về sự khác biệt trung bình giữa các nhóm này")
# Tukey HSD
tukey = pairwise_tukeyhsd(endog=new_DF['tension_strenght'], 
                          groups=new_DF['nozzle_temperature_2'], 
                          alpha=0.05)

tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])


tukey_results["Comparison"] = tukey_results["group1"] + " - " + tukey_results["group2"]


fig, ax = plt.subplots(figsize=(10, 6))


for i, row in tukey_results.iterrows():
    ax.plot([row['lower'], row['upper']], [i, i], 'k-', lw=2)
    ax.plot(row['meandiff'], i, 'ko')  


ax.axvline(0, color='gray', linestyle='dashed')


ax.set_yticks(range(len(tukey_results)))
ax.set_yticklabels(tukey_results["Comparison"])
ax.set_xlabel("Differences in mean levels of as.factor(nozzle_temperature_2)")
ax.set_title("95% family-wise confidence level")

plt.tight_layout()
plt.show()

 
summary_new_DF = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]) 


print("Kết luận từ kiểm định Tukey HSD:")
for idx, row in summary_new_DF.iterrows():
    group1 = row['group1']
    group2 = row['group2']
    meandiff = row['meandiff']
    lower = row['lower']
    upper = row['upper']
    reject = row['reject']  

    if lower < 0 < upper:
        print(f"  - Không có sự khác biệt về trung bình giữa {group1} và {group2} (khoảng tin cậy [{lower:.2f}, {upper:.2f}] chứa 0).")
    else:
        if meandiff > 0:
            print(f"  - Có sự khác biệt về trung bình giữa {group1} và {group2}. Trung bình của {group1} cao hơn {group2} (khoảng tin cậy [{lower:.2f}, {upper:.2f}]).")
        else:
            print(f"  - Có sự khác biệt về trung bình giữa {group1} và {group2}. Trung bình của {group2} cao hơn {group1} (khoảng tin cậy [{lower:.2f}, {upper:.2f}]).")

# =============================================================
# XÂY DỰNG MÔ HÌNH HỒI QUY ĐA BIẾN
# =============================================================
print("\n" + "="*70)
print("PHẦN V. HỒI QUY TUYẾN TÍNH ĐA BIẾN ")
print("="*70)

# KIỂM TRA TƯƠNG QUAN BẰNG HEATMAP
plt.figure(figsize=(10, 8))
corr = new_DF_encoded.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True,
            cbar=True, linewidths=0.5, linecolor='gray')
plt.title("Correlation Heatmap of Variables")
plt.tight_layout()
plt.show()

# Hàm thêm ký hiệu ý nghĩa thống kê
def get_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ' '

# MODEL 1: mô hình đầy đủ trừ fan_speed
model1_formula = "tension_strenght ~ layer_height + wall_thickness + infill_density + infill_pattern + nozzle_temperature + bed_temperature + print_speed + material"
model1 = smf.ols(formula=model1_formula, data=new_DF_encoded).fit()
print('\nMÔ HÌNH 1:')
summary1 = model1.summary2().tables[1]
summary1 = summary1[['Coef.', 'Std.Err.', 't', 'P>|t|']]  # Chọn cột cần
summary1['Signif.'] = summary1['P>|t|'].apply(get_significance)
print(tabulate(summary1,  headers='keys', tablefmt='pretty'))
print(f"R-squared           : {model1.rsquared:.4f}")
print(f"Adjusted R-squared : {model1.rsquared_adj:.4f}")
print(f"F-statistic         : {model1.fvalue:.2f}")
print(f"Prob (F-statistic) : {model1.f_pvalue:.4g}")
print(f"Residual Std Error  : {model1.mse_resid**0.5:.4f}")


# MODEL 2: loại bỏ infill_pattern và print_speed
model2_formula = "tension_strenght ~ layer_height + wall_thickness + infill_density + nozzle_temperature + bed_temperature + material"
model2 = smf.ols(formula=model2_formula, data=new_DF_encoded).fit()
print('\n\nMÔ HÌNH 2:')
summary2 = model2.summary2().tables[1]
summary2 = summary2[['Coef.', 'Std.Err.', 't', 'P>|t|']]
summary2['Signif.'] = summary2['P>|t|'].apply(get_significance)
print(tabulate(summary2,  headers='keys', tablefmt='pretty'))
print(f"R-squared           : {model2.rsquared:.4f}")
print(f"Adjusted R-squared : {model2.rsquared_adj:.4f}")
print(f"F-statistic         : {model2.fvalue:.2f}")
print(f"Prob (F-statistic) : {model2.f_pvalue:.4g}")
print(f"Residual Std Error  : {model2.mse_resid**0.5:.4f}")

# So sánh R-squared và Adjusted R-squared
comparison_table = pd.DataFrame({
    "Model": ["Model 1 (đầy đủ)", "Model 2 (rút gọn)"],
    "R_squared": [model1.rsquared, model2.rsquared],
    "Adj_R_squared": [model1.rsquared_adj, model2.rsquared_adj],
    "F_statistic": [model1.fvalue, model2.fvalue],
    "F_p_value": [model1.f_pvalue, model2.f_pvalue]
})
print("\n🔸 BẢNG SO SÁNH R², R² hiệu chỉnh, F và p-value:")
print(comparison_table)

# Kiểm định F để so sánh 2 mô hình
anova_result = anova_lm(model2, model1)
print("\n🔸 KẾT QUẢ KIỂM ĐỊNH F (Model 2 vs Model 1):")
print(anova_result)
anova_pval = anova_result['Pr(>F)'].iloc[1]
print('p-value=',anova_pval)
# Phân tích p-value từ kiểm định F

if anova_pval > 0.05:
    print(f"\n→ Vì p-value = {anova_pval:.4f} > 0.05 nên KHÔNG bác bỏ H0.")
    print("⇒ Model 2 đơn giản hơn mà không làm giảm đáng kể khả năng giải thích. Chọn Model 2.")
else:
    print(f"\n→ Vì p-value = {anova_pval:.4f} < 0.05 nên bác bỏ H0.")
    print("⇒ Model 1 tốt hơn về mặt thống kê.")

# =============================================================
# KIỂM TRA GIẢ ĐỊNH CỦA MÔ HÌNH 2 (residuals)
# =============================================================
print("\n" + "="*70)
print("KIỂM TRA GIẢ ĐỊNH CỦA MÔ HÌNH HỒI QUY")

# Lấy residuals và fitted values từ model2
influence = OLSInfluence(model2)
residuals = model2.resid
fitted = model2.fittedvalues
standardized_residuals = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag
cooks_distances = influence.cooks_distance[0]

# Tạo figure cho tất cả biểu đồ
plt.figure(figsize=(15, 12))

# 1. Residuals vs Fitted
plt.subplot(2, 2, 1)
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# 2. Q-Q plot
plt.subplot(2, 2, 2)
sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
plt.title('Normal Q-Q')

# 3. Scale-Location
plt.subplot(2, 2, 3)
plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.6)
sns.regplot(x=fitted, y=np.sqrt(np.abs(residuals)), scatter=False, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Scale-Location')

# 4. Residuals vs Leverage
plt.subplot(2, 2, 4)
plt.scatter(leverage, standardized_residuals, s=50, edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='gray', linestyle='--')

# Tính toán Cook's Distance cho từng quan sát
MSE = np.mean(residuals ** 2)  # Sai số bình phương trung bình
p = model2.params.shape[0]  # Số tham số trong mô hình

# Tính toán Cook's Distance cho từng quan sát
cooks_distances = (residuals**2) / (p * MSE) * (leverage) / ((1 - leverage)**2)

# Tạo một giá trị leverage từ min tới max để vẽ curve Cook's Distance
leverage_vals = np.linspace(0.001, max(leverage) - 0.001, 100)

# Tính toán Cook's Distance cho mỗi giá trị leverage
cooks_curve = (leverage_vals * (1 - leverage_vals)) * (max(cooks_distances))  # Công thức Cook's Distance curve

# Vẽ đường cong Cook's Distance
plt.plot(leverage_vals, cooks_curve, color='red', linestyle='-', label="Cook's Distance")

# Ngưỡng Cook's Distance (4/n)
n = len(residuals)
threshold_cooks = 4 / n
# Đánh dấu các quan sát có Cook's Distance lớn hơn ngưỡng
for i in range(len(leverage)):
    if cooks_distances[i] > threshold_cooks:
        plt.annotate(f'{i + 1}', (leverage[i], standardized_residuals[i]),
                     textcoords="offset points", xytext=(0, 5), ha='center')

plt.xlabel('Leverage')
plt.ylabel('Standardized Residuals')
plt.title('Residuals vs Leverage')
plt.legend()
plt.tight_layout()
plt.show()