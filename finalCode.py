import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import csv 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import math
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from scipy.stats import pearsonr

#reading data from the csv file
data = pd.read_csv('TestData.csv')

#creating monthly count of calls
labels, values = zip(*Counter(data["Month"]).items())
MonthlyVolumeData = list(zip(labels, values))


#exporting data to csv file
with open('monthlyCallVolume.csv', mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['Month', 'CallVolume'])

    for row in MonthlyVolumeData:
        writer.writerow(row)


#plotting the monthly call data
plt.bar(labels, values, color ='maroon',
        width = 0.4)

plt.xlabel("Month")
plt.title("Monthly Call Data")
plt.ylabel("Volume of call")
plt.show()



#creating the weekday count for calls
labels, values = zip(*Counter(data["WeekDay"]).items())


#plotting the weekday data
plt.bar(labels, values, color ='maroon',
        width = 0.4)

plt.xlabel("Week day")
plt.title("Week day Call Data")
plt.ylabel("Volume of call")
plt.show()


#exporting weekday data to csv file
WeekDayVolumeData = list(zip(labels, values))
with open('weekDayCallVolume.csv', mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['WeekDay', 'CallVolume'])

    for row in WeekDayVolumeData:
        writer.writerow(row)


#creating weekly data
weekdata = (data["DataWeekNum"].value_counts())
labels, values = zip(*Counter(data["DataWeekNum"]).items())

#plotting weekly data
plt.scatter(labels, values)
z = np.polyfit(labels, values, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(labels, p(labels), color = "black")
plt.title("Raw Data")
plt.xlabel("Week number")
plt.ylabel("Number of calls in the week")
plt.show()


# fill in missing labels and values with the mean
meanValue = int(np.mean(values))
for i in range(1, 303):
    if i not in labels:
        labels = labels + (i,)
        values = values + (meanValue,)


#creating weekly data then exporting values
WeeklyVolumeData = list(zip(labels, values))

with open('weeklyCallVolume.csv', mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['WeekNum', 'CallVolume'])

    for row in WeeklyVolumeData:
        writer.writerow(row)


#Countinuing the forecasting with the weekly data csv file created
df =  pd.read_csv('weeklyCallVolume.csv')

#labeling variables
X = df["WeekNum"]
Y = df["CallVolume"]


#calcualting correlation coefficent and its significance
correlationCoefficient, p_value = pearsonr(X, Y)
print("CorrelationCoefficent :", correlationCoefficient, "p-value: ", p_value )

# adding new column to have a non-zero y-intercept
X = sm.add_constant(X)

#create regression model
model = sm.OLS(Y,X).fit()
print(model.summary())

#Tests for the regression model
print("F-test: ", model.fvalue,  "p-value: " ,model.f_pvalue)
print("T-test: ", model.tvalues, "p-value: ", model.pvalues)
conf_int = model.conf_int(alpha = 0.05)
print("Intercept: ", conf_int, "Slope: ", conf_int)


#predicting the value for upcoming weeks
upComingWeeks = [303, 304, 305, 306, 307, 308, 309, 310]
upComingWeeks = sm.add_constant(upComingWeeks)
simpleRegPrediction = model.predict(upComingWeeks).astype(int)
print("Simple Regression Predictions: ", simpleRegPrediction)

#plotting simple regression predictions
plt.plot(Y)
plt.title("Simple Regression Predictions")
plt.plot(upComingWeeks, simpleRegPrediction, color='red')
plt.show()



# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(df['CallVolume'], model='additive', period=52)


# Ploting the decompositions 
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('CallVolume', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#creating mul model model
'''
model = ExponentialSmoothing(df['CallVolume'],trend='mul',seasonal='mul',seasonal_periods=52).fit()
predictions = model.forecast(8).astype(int)
print("Multiplicative Decomposition Predictions:", predictions)'''


#creating add model
model = ExponentialSmoothing(df['CallVolume'],trend='add',seasonal='add',seasonal_periods=52).fit()
print(model.summary())
predictions = model.forecast(8).astype(int)
print("Additive Decomposition Predictions:", predictions)

#plotting the Holt Winters‘s Trend and Seasonality Method
df['CallVolume'].plot(legend=True,label='Trainning')
predictions.plot(legend=True,label='prediction')
plt.title('Holt Winters ADD')
plt.show()


#Plotting ACF
plot_acf(Y, lags = 52)
plt.show()

#box pierce test which also gives alternative test results
boxPierce = sm.stats.acorr_ljungbox(Y, lags=52, boxpierce=True)
# Print the results
print("Box-Pierce statistic for lags 1 to 52: ", boxPierce)

#plotting pacf
plot_pacf(Y, lags=52,method='ywm')
plt.show()

#checking stationary
result = adfuller(Y)


# Test statistics p-value, and other test results
testStats = result[0]
p_value = result[1]
criticalVal = result[4]


# Check if the test statistic is less than the critical value at the 1% significance level
if testStats < criticalVal['1%']:
    print('Stationary with 99% confidence')
elif testStats < criticalVal['5%']:
    print('Stationary with 95% confidence')
elif testStats < criticalVal['10%']:
    print('Stationary with 90% confidence')
else:
    print('Not stationary')


#performing seasonal diff
#seasonalDiff = Y.diff(periods=52)

#first 52 in Nan so removing those values
#seasonalDiff = seasonalDiff[52:]

#finding the best arima model
ArimaModel = auto_arima(Y, seasonal=True, m=52, max_p=5, max_d=2, max_q=5)
print(ArimaModel.summary())
ArimaModel.fit(Y)



# generate predictions on the test data
predictions = ArimaModel.predict(n_periods= 8).astype(int)
predictions = np.clip(predictions, 0, None)
print("Arima predictions: ", predictions)
plt.title("ARIMA Prediction")
plt.plot(Y)
plt.plot(predictions, color='red')
plt.show()