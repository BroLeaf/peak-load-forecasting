import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# read data set
data = pd.read_csv("2017.1.1 ~ 2019.1.31 的電力供需.csv")
data = data[["日期","淨尖峰供電能力(MW)","尖峰負載(MW)","備轉容量(MW)","備轉容量率(%)"]]
# remove error data
data = data.reset_index()
np.isnan(data).any()
data.dropna(inplace=True)

X = data[["日期","淨尖峰供電能力(MW)","備轉容量(MW)","備轉容量率(%)"]]
y = data[["尖峰負載(MW)"]]


# 75% 劃分為 train set, 25%劃分為 test set
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

# train module
linreg = LinearRegression()
linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)

print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
