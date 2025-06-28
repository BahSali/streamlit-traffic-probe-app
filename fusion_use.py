import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import random

class CNNTimeSeries(nn.Module):
    def __init__(self, num_features, lookback):
        super(CNNTimeSeries, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten_size = self._get_flatten_size(num_features, lookback)
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, lookback)

    def _get_flatten_size(self, num_features, lookback):
        x = torch.zeros(1, num_features, lookback)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
        
def main():
    # Load model
    # model = torch.load('full_model_new_loss.pth')
    # model = torch.load('full_model_new_loss.pth', weights_only=False)
    model = torch.load('full_model_new_loss.pth', map_location='cpu', weights_only=False)

    
    model.eval()
    print("Model loaded successfully!")
    
    # Load data
    df_raw = pd.read_csv('STIB_speeds.csv', sep=',', encoding='latin1')
    df_raw.columns = df_raw.columns.str.strip()
    df = df_raw.copy()
    
    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['minute'] = df['Time'].dt.minute
    df['dayofweek'] = df['Time'].dt.dayofweek
    df = df.sort_values('Time').reset_index(drop=True)
    df['name'] = df['SegmentID'].astype('category').cat.codes
    
    df['STIB'] = pd.to_numeric(df['Speed'].astype(str).str.replace(',', '.'), errors='coerce')
    stib_orig = df['STIB'].copy()
    
    features = ['STIB', 'hour', 'minute', 'dayofweek', 'name']
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])
    scaler_y.fit(df_raw[['Speed']].dropna())
    
    def create_sequences_ignore_nan(df, features, lookback):
        sequences = []
        last_valid_inputs = []
        index_list = []
        for seg_id in df['SegmentID'].unique():
            sub = df[df['SegmentID'] == seg_id].reset_index()
            values = sub[features].values
            stib_vals = sub['STIB'].values
            for i in range(len(sub)):
                start = max(0, i - lookback + 1)
                window = values[start:i+1]
                window_valid = window[~np.isnan(window).any(axis=1)]
                if window_valid.shape[0] == 0:
                    continue
                if window_valid.shape[0] < lookback:
                    pad = np.tile(window_valid[-1], (lookback - window_valid.shape[0], 1))
                    window_valid = np.vstack([window_valid, pad])
                sequences.append(window_valid)
                last_valid_inputs.append(stib_vals[i])
                index_list.append(sub['index'][i])
        if len(sequences) == 0:
            return np.zeros((0, lookback, len(features))), np.array([]), np.array([])
        return np.stack(sequences), np.array(last_valid_inputs), np.array(index_list)
    
    lookback = 10
    X, input_stib_last, row_indices = create_sequences_ignore_nan(df, features, lookback)
    print(f"X shape: {X.shape}")
    
    X_t = torch.tensor(X, dtype=torch.float32).permute(0,2,1)
    with torch.no_grad():
        preds_norm = model(X_t).numpy()  # (n_samples, lookback)
    
    # Get model output for the last timestep, then inverse-transform
    pred_scalar = preds_norm[:, -1].reshape(-1,1)
    preds_inv = scaler_y.inverse_transform(pred_scalar).ravel()
    
    # Limit predictions to [Speed-6, Speed+6] and strictly positive
    df_out = df_raw.copy()
    df_out['Prediction'] = pd.NA
    for idx, pred in zip(row_indices, preds_inv):
        orig_speed = df_out.at[idx, 'Speed']
        if pd.isna(orig_speed):
            df_out.at[idx, 'Prediction'] = max(pred, 0.01)
        else:
            lower = max(orig_speed - 6, 0.01)
            upper = orig_speed + 6
            clipped = min(max(pred, lower), upper)
            df_out.at[idx, 'Prediction'] = clipped
    
    df_out['Prediction'] = df_out['Prediction'].fillna(0.01)
    df_out['Prediction'] = df_out['Prediction'].astype(float)
    
    # Add StreetName if you have the segment file
    try:
        seg_file = 'Etterbeek_STIB_segments.csv'
        df_seg  = pd.read_csv(seg_file, sep=';', encoding='latin1')
        df_seg.columns = df_seg.columns.str.strip()
        df_seg = df_seg.rename(columns={'Name - start': 'Name-start', 'Name - stop': 'Name-stop'})
        df_seg['StreetName'] = df_seg['Name-start'] + ' / ' + df_seg['Name-stop']
        df_seg_unique = df_seg.drop_duplicates(subset='ID_graph_edge', keep='first')
        street_map = df_seg_unique.set_index('ID_graph_edge')['StreetName']
        df_out['StreetName'] = df_out['SegmentID'].map(street_map)
    except Exception as e:
        print("Could not add StreetName column:", e)
    
    df_out = df_out.sort_values(by=['SegmentID', 'Time']).reset_index(drop=True)
    
    # Only keep latest timestamp per segment for final results (optional)
    latest_time = df_out['Time'].max()
    df_final = df_out[df_out['Time'] == latest_time].copy()
    
    # ========== CONSTRAINTS ON df_final ==========
    for i, row in df_final.iterrows():
        speed = row['Speed']
        pred = row['Prediction']
        need_random = False
    
        # 1. If speed > 50, re-estimate (here: set random between 11 and 43)
        if pd.notnull(speed) and speed > 50:
            df_final.at[i, 'Prediction'] = random.randint(11, 43)
            print("*** Speed > 50: replaced prediction with random value for index", i)
            need_random = True
    
        # 2. If speed is null, zero or negative: set random [11,43] and print ***
        if pd.isnull(speed) or speed <= 0:
            df_final.at[i, 'Prediction'] = random.randint(11, 43)
            print(speed)
            print("*** Speed is null/0/negative: replaced prediction with random value for index", i)
            need_random = True
    
    # ========== SAVE ==========
    output_file = 'results.csv'
    df_final.to_csv(output_file, sep=';', encoding='latin1', index=False)
    print(f"\nSaved {output_file} with predictions!")
