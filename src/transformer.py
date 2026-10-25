#!/usr/bin/env python3
"""
Optimized Time Series Transformer for Multivariate Prediction Tasks.

This script converts a standard CSV file into a supervised learning format.
It creates a sliding window containing:
- Past values (Lags) for all variables (t-ph to t-1).
- Current values (t0).
- Future values (Horizons) for the target variable (t+1 to t+fh).
"""

import pandas as pd
from tqdm import tqdm
from .logger_setup import setup_framework_logger

logger = setup_framework_logger(__name__)
def transform_time_series(input_file, output_file, fv, fh, ph, time_col=None):
    """
    Transforms time series data for regression tasks with temporal alignment.
    
    Args:
        input_file (str): Path to input CSV.
        output_file (str): Path to save the transformed CSV.
        fv (int): Forecast Variable index (target).
        fh (int): Forecast Horizon (number of future steps).
        ph (int): Past History (number of past lag steps).
        time_col (str): Name of the timestamp column.
    """
    logger.info(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)

    # Sort data by time to ensure valid lag generation
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(time_col).reset_index(drop=True)

    original_columns = df.columns.tolist()
    if fv < 0 or fv >= len(original_columns):
        raise ValueError(f"Target index {fv} is out of range.")

    target_name = original_columns[fv]
    if time_col is not None and target_name == time_col:
        raise ValueError("The time column cannot be used as the forecast target.")

    timestamps = None

    # Handle time column separately to avoid using it as a predictive feature
    if time_col is not None and time_col in df.columns:
        timestamps = df[time_col].copy()
        df = df.drop(columns=[time_col])

    columns = df.columns.tolist()
    target_index = columns.index(target_name)
    logger.info(f"Target variable: {target_name} (adjusted index: {target_index})")

    transformed_rows = []
    output_timestamps = []

    # Sliding window logic: [t-ph, ..., t-1, t0, t+1, ..., t+fh]
    logger.info("Generating sliding windows...")
    for i in tqdm(range(len(df) - (ph + fh))):
        row = []
        
        # 1. Past context: All variables from t-ph to t-1
        if ph != 0:
            past_window = df.iloc[i : i + ph].values
            for step in range(ph):
                row.extend(past_window[step])

        # 2. Present state: Values of all variables at t0
        current_state = df.iloc[i + ph].values
        row.extend(current_state)

        # 3. Future horizon: Target variable values from t+1 to t+fh
        if fh != 0:
            future_target_values = df.iloc[i + ph + 1 : i + ph + fh + 1, target_index].values
            row.extend(future_target_values)

        transformed_rows.append(row)

        # Map the timestamp to the "present" moment (t0)
        if timestamps is not None:
            output_timestamps.append(timestamps.iloc[i + ph])

    # Constructing descriptive column names
    new_column_names = []
    
    # Names for Lags (t-ph to t-1)
    for t in range(ph):
        lag_step = ph - t
        for col in columns:
            new_column_names.append(f"{col}_t-{lag_step}")
            
    # Names for Present (t0)
    new_column_names.extend(columns)
    
    # Names for Horizons (t+1 to t+fh)
    for t in range(fh):
        new_column_names.append(f"{target_name}_t+{t+1}")

    # Create the final DataFrame
    transformed_df = pd.DataFrame(transformed_rows, columns=new_column_names)

    # Re-insert the time column at the first position
    if timestamps is not None:
        transformed_df.insert(0, time_col, output_timestamps)

    # Save results
    transformed_df.to_csv(output_file, index=False)
    logger.info(f"Transformation complete. Samples created: {len(transformed_df)}")
    logger.info(f"File saved at: {output_file}")
