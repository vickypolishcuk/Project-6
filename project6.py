import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import VAR

import warnings
warnings.filterwarnings('ignore')

# Завантаження даних
df = pd.read_csv('Download Data - STOCK_US_XNYS_CSV.csv', parse_dates=True, index_col='Date')
print("Вхідні дані:\n", df)

# Припустимо, що у нас є DataFrame з іменем df
df.plot()
plt.title('Вхідні дані')
plt.show()

# Заміна ком в числа
df['Volume'] = df['Volume'].str.replace(',', '.')
# Перетворення колонки 'Volume' у числовий тип даних
df['Volume'] = pd.to_numeric(df['Volume'])

# Налаштування моделі ARIMA
model = ARIMA(df['Volume'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Прогнозування Volume
forecast = model_fit.forecast(steps=10)
print("Прогнозування Volume:\n", forecast)

model2 = VAR(df)
results = model2.fit()
print(results.summary())

# Прогнозування по параметру 'Open'
model_open = ARIMA(df['Open'], order=(10, 1, 1))
model_fit_open = model_open.fit()
forecast_open = model_fit_open.forecast(steps=10)

# Прогнозування по параметру 'High'
model_high = ARIMA(df['High'], order=(10, 1, 1))
model_fit_high = model_high.fit()
forecast_high = model_fit_high.forecast(steps=10)

# Прогнозування по параметру 'Low'
model_low = ARIMA(df['Low'], order=(10, 1, 1))
model_fit_low = model_low.fit()
forecast_low = model_fit_low.forecast(steps=10)

# Прогнозування по параметру 'Close'
model_close = ARIMA(df['Close'], order=(10, 1, 1))
model_fit_close = model_close.fit()
forecast_close = model_fit_close.forecast(steps=10)

# Створення DataFrame для прогнозованих значень
forecast_df = pd.DataFrame({
    'Open': forecast_open,
    'High': forecast_high,
    'Low': forecast_low,
    'Close': forecast_close
})
print("Прогнозовані решта значень:")
print(forecast_df)

# Виведення графіку
forecast_df.plot()
plt.legend(loc='best')
plt.title('Прогнозовані дані')
plt.show()


# Аналіз тренду
rolling_mean = df['Volume'].rolling(window=10, min_periods=1).mean()
plt.plot(df['Volume'], color='blue', label='Оригінальний ряд')
plt.plot(rolling_mean, color='red', label='Середнє')
plt.legend(loc='best')
plt.title('Середнє для оригінального ряда (Volume)')
plt.xticks(rotation=15)
plt.show()


# Аналіз сезонності
result = seasonal_decompose(df['Volume'], model='multiplicative', period=7)
result.plot()
plt.xticks(rotation=15)
plt.show()


# Перевірка стаціонарності
result = adfuller(df['Volume'])
print('ADF статистика: %f' % result[0])
print('p-value: %f' % result[1])

# Оптимальне диференціювання
print("Оптимальне диференціювання: ", ndiffs(df['Volume'], test='adf'))


# Аналіз залишків
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title("Аналіз залишків")
plt.show()

# Перевірка на автокореляцію
plot_acf(residuals, lags=20)
plt.title("Автокореляція")
plt.show()
