import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, TimeDistributed, Multiply, Softmax, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration
config = {
    'sequence_length': 60,
    'val_size': 100,
    'test_size': 200,
    'epochs': 50,
    'batch_size': 32,
    'random_state': 42
}

# === 1. Load data and compute indicators ===
try:
    data = yf.download('000001.SS', start='2010-01-01', end='2025-04-21', auto_adjust=True)
    if data.empty:
        raise ValueError("No data downloaded")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit(1)


data = data[~data.index.duplicated(keep='first')]
data = data.dropna()

data['return'] = data['Close'].pct_change()
data['target_return'] = data['return'].shift(-1)

data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
data['Volatility_5'] = data['return'].rolling(window=5).std()

data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'return',
            'SMA_5', 'SMA_10', 'Momentum_5', 'Volatility_5']
target = data['target_return'].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features])


target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()


sequence_length = config['sequence_length']
X, y, close_base = [], [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(target_scaled[i+sequence_length])
    close_base.append(data['Close'].values[i + sequence_length - 1])

X = np.array(X)
y = np.array(y).flatten()  
close_base = np.array(close_base).flatten()  

# === 2. Split Data ===
val_size = config['val_size']
test_size = config['test_size']
train_size = len(X) - val_size - test_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[-val_size:], y[-val_size:]
X_test, y_test = X[train_size:train_size+test_size], y[train_size:train_size+test_size]
close_base_test = close_base[train_size:train_size+test_size]

y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
true_close = close_base_test * (1 + y_test_unscaled)


print("Shape of X_train:", X_train.shape)
print("Shape of close_base_test:", close_base_test.shape)
print("Shape of y_test:", y_test.shape)
print("Shape of true_close:", true_close.shape)

# === 3. CNN-BiLSTM-Attention Model ===
input_layer = Input(shape=(sequence_length, X.shape[2]))
x = Conv1D(32, 3, activation='relu')(input_layer)
x = MaxPooling1D(2)(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(50, return_sequences=True))(x)
attention = TimeDistributed(Dense(1))(x)
attention = Softmax(axis=1)(attention)
context = Multiply()([x, attention])
context = Lambda(lambda x: K.sum(x, axis=1))(context)
output = Dense(1)(context)

model_dl = Model(inputs=input_layer, outputs=output)
model_dl.compile(optimizer='adam', loss='mse')


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_dl.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config['epochs'],
             batch_size=config['batch_size'], callbacks=[early_stopping], verbose=1)

y_pred_dl_scaled = model_dl.predict(X_test).flatten()
y_pred_dl = target_scaler.inverse_transform(y_pred_dl_scaled.reshape(-1, 1)).flatten()

# === 4. Transformer Model (PyTorch) ===
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, horizon):
        super().__init__()
        self.horizon = horizon
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, horizon)

    def forward(self, src):
        x = self.input_linear(src)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_linear(x)
        return x.view(x.size(0), self.horizon, 1)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train[:, None, None], dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=config['batch_size'], shuffle=True)

X_val_torch = torch.tensor(X_val, dtype=torch.float32)
y_val_torch = torch.tensor(y_val[:, None, None], dtype=torch.float32)

model_tf = TransformerRegressor(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2,
                                dim_feedforward=128, horizon=1)
optimizer = torch.optim.Adam(model_tf.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(config['epochs']):
    model_tf.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model_tf(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_tf.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model_tf.eval()
    with torch.no_grad():
        val_pred = model_tf(X_val_torch)
        val_loss = loss_fn(val_pred, y_val_torch)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}")

model_tf.eval()
with torch.no_grad():
    tf_pred_scaled = model_tf(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).squeeze(-1).numpy()
tf_pred = target_scaler.inverse_transform(tf_pred_scaled.reshape(-1, 1)).flatten()

# === 5. Random Forest ===
X_rf_train = X_train.reshape(X_train.shape[0], -1)
X_rf_test = X_test.reshape(X_test.shape[0], -1)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=config['random_state'])
rf.fit(X_rf_train, y_train)

y_pred_rf_scaled = rf.predict(X_rf_test)
y_pred_rf = target_scaler.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).flatten()

# === 6. Ensemble ===
y_val_unscaled = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_pred_dl_val = target_scaler.inverse_transform(model_dl.predict(X_val).reshape(-1, 1)).flatten()
with torch.no_grad():
    tf_pred_val = target_scaler.inverse_transform(model_tf(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1).squeeze(-1).numpy().reshape(-1, 1)).flatten()
y_pred_rf_val = target_scaler.inverse_transform(rf.predict(X_val.reshape(X_val.shape[0], -1)).reshape(-1, 1)).flatten()

weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_mape, best_weights = float('inf'), None
close_base_val = close_base[-val_size:]

for w_dl, w_tf in product(weights, weights):
    w_rf = 1 - w_dl - w_tf
    if w_rf < 0 or w_rf > 1:
        continue
    ensemble_val = w_dl * y_pred_dl_val + w_tf * tf_pred_val + w_rf * y_pred_rf_val
    val_close = close_base_val * (1 + ensemble_val)
    mape = mean_absolute_percentage_error(close_base_val * (1 + y_val_unscaled), val_close)
    if mape < best_mape:
        best_mape, best_weights = mape, (w_dl, w_tf, w_rf)

alpha_dl, alpha_tf, alpha_rf = best_weights
print(f"Optimized weights - CNN-BiLSTM: {alpha_dl:.2f}, Transformer: {alpha_tf:.2f}, Random Forest: {alpha_rf:.2f}")


ensemble_return = alpha_dl * y_pred_dl + alpha_tf * tf_pred + alpha_rf * y_pred_rf
ensemble_close = close_base_test * (1 + ensemble_return)

# === 7. Evaluation ===
# Evaluate ensemble
mape = mean_absolute_percentage_error(true_close, ensemble_close)
rmse = np.sqrt(mean_squared_error(true_close, ensemble_close))
r2 = r2_score(true_close, ensemble_close)

print("\n📊 Ensemble with Technical Indicators")
print(f"MAPE: {mape:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Evaluate individual models
for name, pred in [('CNN-BiLSTM', y_pred_dl), ('Transformer', tf_pred), ('Random Forest', y_pred_rf)]:
    pred_close = close_base_test * (1 + pred)
    print(f"\n{name}:")
    print(f"MAPE: {mean_absolute_percentage_error(true_close, pred_close):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(true_close, pred_close)):.4f}")
    print(f"R²   : {r2_score(true_close, pred_close):.4f}")

# === 8. Plot ===
plt.clf()
plt.figure(figsize=(14, 7))
plt.plot(data.index[train_size + sequence_length:train_size + sequence_length + len(y_test)],
         true_close, label='Actual Close Price', color='blue')
plt.plot(data.index[train_size + sequence_length:train_size + sequence_length + len(y_test)],
         ensemble_close, label='Predicted Close (Ensemble)', linestyle='--', color='green')
plt.title("Actual vs Predicted SPY Close (Ensemble with Indicators)", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Price (USD)", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
plt.savefig('spy_prediction_plot.png')