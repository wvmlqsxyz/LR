
# 3. OLS 회귀분석 실행 및 결과 출력
results = sm.OLS(y, x).fit()
print(results.summary())

# -------------------------------------------------------
# VIF
# -------------------------------------------------------
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print("\n=== VIF(다중공선성) 결과 ===")
print(vif_data.sort_values(by="VIF", ascending=False))

# 4. 시각화 (산점도 및 회귀선)
independent_vars_to_plot = [col for col in x_cols if col != 'const'] # 상수항 제외하고 그리기

for col in independent_vars_to_plot:
    plt.figure(figsize=(8, 5))
    sns.regplot(x=data[col], y=y, data=data, 
                scatter_kws={'alpha':0.3},    
                line_kws={'color':'red', 'linestyle':'--'}) 
    plt.xlabel(col, fontsize=12)
    plt.ylabel('매출', fontsize=12)
    plt.title(f'{col} vs. Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 5. 잔차 분석 (모형 적합성 판단)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=results.fittedvalues, y=results.resid, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted', fontsize=15)
plt.show()
