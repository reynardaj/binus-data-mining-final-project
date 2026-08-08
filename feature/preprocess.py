import os
import glob
import pandas as pd
import numpy as np

# Column mapping from Korean telemetry names to English standard feature names
COLUMN_MAP = {
    'DAT': 'DAT',
    'obs_time': 'obs_time',
    '내부온도관측치': 'temp',
    '내부습도관측치': 'humidity',
    'co2관측치': 'co2',
    'ec관측치': 'ec',
    '시간당분무량': 'water_spray',
    '일간누적분무량': 'cum_water_spray_raw',
    '시간당백색광량': 'white_light',
    '일간누적백색광량': 'cum_white_light_raw',
    '시간당적색광량': 'red_light',
    '일간누적적색광량': 'cum_red_light_raw',
    '시간당청색광량': 'blue_light',
    '일간누적청색광량': 'cum_blue_light_raw',
    '시간당총광량': 'total_light',
    '일간누적총광량': 'cum_total_light_raw'
}

def apply_kalman_filter(series, process_variance=1e-3, measurement_variance=1e-1):
    """
    1D Kalman Filter implementation for noise reduction on daily telemetry features.
    """
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    
    # Initialization
    val_0 = series.iloc[0] if not pd.isna(series.iloc[0]) else 0.0
    xhat[0] = val_0
    P[0] = 1.0
    
    for k in range(1, n):
        # Time update (Predict)
        xhatminus = xhat[k-1]
        Pminus = P[k-1] + process_variance
        
        # Measurement update (Correct)
        z = series.iloc[k]
        if pd.isna(z):
            z = xhatminus
        K = Pminus / (Pminus + measurement_variance)
        xhat[k] = xhatminus + K * (z - xhatminus)
        P[k] = (1 - K) * Pminus
        
    return pd.Series(xhat, index=series.index)

def preprocess_and_engineer(data_dir="data"):
    """
    KDD Phase 1 & 2: Data Integration, Cleaning, Chronological Alignment,
    and Feature Engineering.
    """
    train_input_dir = os.path.join(data_dir, "train_input")
    train_target_dir = os.path.join(data_dir, "train_target")
    
    input_files = sorted(glob.glob(os.path.join(train_input_dir, "CASE_*.csv")))
    if not input_files:
        raise FileNotFoundError(f"No CASE CSV files found in {train_input_dir}")
        
    all_cases_df_list = []
    
    for input_file in input_files:
        case_id = os.path.basename(input_file).split('.')[0]
        target_file = os.path.join(train_target_dir, f"{case_id}.csv")
        
        if not os.path.exists(target_file):
            print(f"Warning: Target file missing for {case_id}, skipping.")
            continue
            
        # Load telemetry input and weight target
        df_in = pd.read_csv(input_file)
        df_target = pd.read_csv(target_file)
        
        # Rename columns
        df_in = df_in.rename(columns=COLUMN_MAP)
        
        # Data Cleaning & Sensor Anomaly Fixes
        df_in = df_in.ffill().bfill()
        df_in['temp'] = df_in['temp'].clip(0.0, 50.0)
        df_in['humidity'] = df_in['humidity'].clip(0.0, 100.0)
        df_in['co2'] = df_in['co2'].clip(0.0, 3000.0)
        df_in['ec'] = df_in['ec'].clip(0.0, 10.0)
        df_in['water_spray'] = df_in['water_spray'].clip(lower=0.0)
        df_in['total_light'] = df_in['total_light'].clip(lower=0.0)
        df_in['white_light'] = df_in['white_light'].clip(lower=0.0)
        df_in['red_light'] = df_in['red_light'].clip(lower=0.0)
        df_in['blue_light'] = df_in['blue_light'].clip(lower=0.0)
        
        # Aggregate hourly records (24 rows per DAT 0..27) to Daily Level
        # Group by DAT in df_in
        daily_records = []
        
        for dat, group in df_in.groupby('DAT'):
            daily_dict = {
                'input_DAT': dat,
                'target_DAT': dat + 1,  # Chronological alignment: DAT d (0..27) predicts DAT d+1 (1..28)
                
                # Daily Averages, Min, Max, Std
                'temp_mean': group['temp'].mean(),
                'temp_std': group['temp'].std(ddof=0),
                'temp_min': group['temp'].min(),
                'temp_max': group['temp'].max(),
                
                'humidity_mean': group['humidity'].mean(),
                'humidity_std': group['humidity'].std(ddof=0),
                'humidity_min': group['humidity'].min(),
                'humidity_max': group['humidity'].max(),
                
                'co2_mean': group['co2'].mean(),
                'co2_std': group['co2'].std(ddof=0),
                'co2_min': group['co2'].min(),
                'co2_max': group['co2'].max(),
                
                'ec_mean': group['ec'].mean(),
                'ec_std': group['ec'].std(ddof=0),
                'ec_min': group['ec'].min(),
                'ec_max': group['ec'].max(),
                
                'water_spray_sum': group['water_spray'].sum(),
                'water_spray_mean': group['water_spray'].mean(),
                'water_spray_max': group['water_spray'].max(),
                
                'total_light_sum': group['total_light'].sum(),
                'total_light_mean': group['total_light'].mean(),
                'total_light_max': group['total_light'].max(),
                
                'white_light_sum': group['white_light'].sum(),
                'red_light_sum': group['red_light'].sum(),
                'blue_light_sum': group['blue_light'].sum()
            }
            daily_records.append(daily_dict)
            
        df_daily = pd.DataFrame(daily_records)
        df_daily['case_id'] = case_id
        
        # Sort by target_DAT
        df_daily = df_daily.sort_values('target_DAT').reset_index(drop=True)
        
        # Cumulative Features over 28-day cycle
        df_daily['cum_water_spray'] = df_daily['water_spray_sum'].cumsum()
        df_daily['cum_total_light'] = df_daily['total_light_sum'].cumsum()
        df_daily['cum_red_light'] = df_daily['red_light_sum'].cumsum()
        df_daily['cum_white_light'] = df_daily['white_light_sum'].cumsum()
        df_daily['cum_co2'] = df_daily['co2_mean'].cumsum()
        
        # Interaction Features
        df_daily['ec_x_spray'] = df_daily['ec_mean'] * df_daily['water_spray_sum']
        df_daily['temp_x_humidity'] = df_daily['temp_mean'] * df_daily['humidity_mean']
        df_daily['light_x_co2'] = df_daily['total_light_sum'] * df_daily['co2_mean']
        df_daily['temp_x_light'] = df_daily['temp_mean'] * df_daily['total_light_sum']
        
        # Advanced Noise Reduction / Smoothing Filters on Daily Averages
        # Technique 1: Exponential Moving Average (EMA)
        df_daily['temp_mean_ema'] = df_daily['temp_mean'].ewm(span=3).mean()
        df_daily['humidity_mean_ema'] = df_daily['humidity_mean'].ewm(span=3).mean()
        df_daily['co2_mean_ema'] = df_daily['co2_mean'].ewm(span=3).mean()
        df_daily['ec_mean_ema'] = df_daily['ec_mean'].ewm(span=3).mean()
        df_daily['total_light_mean_ema'] = df_daily['total_light_mean'].ewm(span=3).mean()
        
        # Technique 2: 1D Kalman Filter Smoothing
        df_daily['temp_mean_kf'] = apply_kalman_filter(df_daily['temp_mean'])
        df_daily['humidity_mean_kf'] = apply_kalman_filter(df_daily['humidity_mean'])
        df_daily['co2_mean_kf'] = apply_kalman_filter(df_daily['co2_mean'])
        df_daily['ec_mean_kf'] = apply_kalman_filter(df_daily['ec_mean'])
        df_daily['total_light_mean_kf'] = apply_kalman_filter(df_daily['total_light_mean'])
        
        # Merge with target weight
        # Match target_DAT in df_daily with DAT in df_target
        df_merged = pd.merge(
            df_daily,
            df_target[['DAT', 'predicted_weight_g']],
            left_on='target_DAT',
            right_on='DAT',
            how='inner'
        )
        df_merged = df_merged.drop(columns=['DAT']).rename(columns={'target_DAT': 'DAT'})
        
        all_cases_df_list.append(df_merged)
        
    final_df = pd.concat(all_cases_df_list, ignore_index=True)
    
    # Save transformed training dataset
    output_path = os.path.join(data_dir, "transformed_train_data.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Transformed training dataset successfully exported to: {output_path}")
    print(f"Shape: {final_df.shape}, Cases: {final_df['case_id'].nunique()}")
    return final_df

if __name__ == "__main__":
    preprocess_and_engineer()
