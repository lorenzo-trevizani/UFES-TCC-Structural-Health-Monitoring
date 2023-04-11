import pandas as pd
import numpy as np

def getStatisticalCaracteristics(original_df,sensor_column_id) :
    """
    Take dataframe and Sensor of choice ('S1') and return the calculated statistical characteristics for group of 50 values
    """
    # Sensor column in dataframe 
    sensor_df = original_df[sensor_column_id]

    # Group the values in S1 column by 50 and calculate statistical characteristics
    grouped_values = sensor_df.groupby(np.arange(len(original_df)) // 50).agg(['mean', 'std', 'min', 'max', 'median', 'skew'])

    # Calculate the root mean square and the difference between max and min for each group
    grouped_values['rms'] = sensor_df.groupby(np.arange(len(original_df)) // 50).apply(lambda x: np.sqrt(np.mean(x**2)))
    grouped_values['amp_max_min'] = sensor_df.groupby(np.arange(len(original_df)) // 50).apply(lambda x: x.max() - x.min())
    grouped_values['kurtosis'] = sensor_df.groupby(np.arange(len(original_df)) // 50).apply(lambda x: pd.Series.kurtosis(x))

    # Reset the index of the grouped_values dataframe
    return grouped_values;
