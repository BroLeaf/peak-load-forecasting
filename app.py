import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# read data set
data = pd.read_csv("data/2017.1.1 ~ 2019.1.31 的電力供需.csv", encoding='utf-8')
data = data[["日期","淨尖峰供電能力(MW)","尖峰負載(MW)","備轉容量(MW)","備轉容量率(%)"]]
# remove error data
data = data.reset_index()
np.isnan(data).any()
data.dropna(inplace=True)

X = data[["日期","淨尖峰供電能力(MW)","備轉容量(MW)","備轉容量率(%)"]]
y = data[["尖峰負載(MW)"]]


# 75% 劃分為 train set, 25%劃分為 test set
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

# train module
linreg = LinearRegression()
linreg.fit(X,y)

# read support data
pre_data = pd.read_csv("data/power_supply_predicted.csv", encoding='utf-8')
pre_data = pre_data.rename(index=str,columns={"日期(年/月/日)":"日期","預估淨尖峰供電能力(萬瓩)":"淨尖峰供電能力(MW)","預估瞬時尖峰負載(萬瓩)":"尖峰負載(MW)","預估尖峰備轉容量率(%)":"備轉容量率(%)","預估尖峰備轉容量(萬瓩)":"備轉容量(MW)"})
# modify format
for i in range(0, 7):
    s=pre_data.at[str(i), "日期"]
    pre_data.at[str(i), "日期"]=s[0:4]+s[5:7]+s[8:10]
    pre_data.at[str(i), "淨尖峰供電能力(MW)"]*=10
    pre_data.at[str(i), "備轉容量(MW)"]*=10
pre_data = pre_data[["日期","淨尖峰供電能力(MW)","備轉容量(MW)","備轉容量率(%)"]]    
# print(pre_data)

# predict future peak load for a week and output
pred = linreg.predict(pre_data)
pred_output = pd.DataFrame(pred, columns=["peak_load(MW)"])
pred_output.insert(0,"date", pre_data["日期"].to_numpy())
print(pred_output)
pred_output.to_csv("submission.csv", encoding='utf-8', index=False)
