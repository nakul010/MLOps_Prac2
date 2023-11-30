import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
data = np.genfromtxt('sample.csv', delimiter=',')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'scaler_object.joblib')

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)

from sklearn.linear_model import LinearRegression
import joblib
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
model = LinearRegression()
model.fit(X_train, y_train)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]
predictions = model.predict(X_test)
joblib.dump(model, 'linear_regression_model.joblib')
loaded_model = joblib.load('linear_regression_model.joblib')
loaded_predictions = loaded_model.predict(X_test)
if np.array_equal(predictions, loaded_predictions):
    print("Output : Predictions match!")
else:
    print("Output : Predictions differ!")