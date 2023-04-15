import pandas as pd
import numpy as np
import re

def initDatabase(path) :
    """
    Take [list] of paths to loop through and return the dataframe unified and merge with damage scenario
    """
    ## Get column names from database
    data = pd.read_csv(path[0], skiprows=10, sep='\t', header=None, dtype='unicode');
    dataframe_initial = pd.DataFrame(data);

    #Quantity of file's columns
    qty_col = len(data.columns);

    #Create array for column's names
    column_label_list = [
        'Time'
    ];

    for j in range(qty_col - 1):
        column_label_list.append('S'+ str(j+1))

    column_label_list.append('Scenario');

    ## Concatenated files
    appended_file = [];

    for i in range(len(path)):
        ## IMPORTING DATABASE TO VARIABLE df
        data = pd.read_csv(path[i], skiprows=11, sep='\t', header=None, dtype='unicode');
        df = pd.DataFrame(data);

        #Quantity of file's rows
        qty_row = len(data[0]);

        #Import scenario for damage
        scenario_number = getDamageScenarioLabel(path[i]);
        scenario_array = np.full((qty_row), scenario_number).tolist();

        #Merge scenario number into dataframe
        df[qty_col] = scenario_array;

        ## Assign df to empty list
        appended_file.append(df);

        #Merge scenario number into dataframe
        dataframe_final = pd.concat(appended_file, axis=0, ignore_index=True, sort=False);
        
    dataframe_final.columns = column_label_list;

    return dataframe_final;

def getDamageScenarioLabel(path) :
    """
    Take path variable .txt and return int associate in the end of file name.
    Exemple:
    input: ./test/banana123.txt
    return: 123
    """

    regex = re.compile(r'\d+')
    if (path[-5] != 'U'):
        scenario_number = [int(x) for x in regex.findall(path)];
    else :
        scenario_number = [0];

    return scenario_number[0];

def getStatisticalCaracteristics(original_df,sensor_column_id) :
    """
    Take dataframe and Sensor of choice ('S1') and return the calculated statistical characteristics for group of 50 values
    """
    # Sensor column in dataframe 
    sensor_df = original_df[sensor_column_id]

    # Group the values in S1 column by 50 and calculate statistical characteristics
    grouped_values = sensor_df.groupby(np.arange(len(original_df)) // 50).agg(['mean', 'std', 'median', 'skew'])

    # Calculate the root mean square and the difference between max and min for each group
    grouped_values['amp_max_min'] = sensor_df.groupby(np.arange(len(original_df)) // 50).apply(lambda x: x.max() - x.min())
    grouped_values['kurtosis'] = sensor_df.groupby(np.arange(len(original_df)) // 50).apply(lambda x: pd.Series.kurtosis(x))

    # Reset the index of the grouped_values dataframe
    return grouped_values;

def createDatabase(path):
    """
    Take one path and return it in a dataframe
    """
    data = pd.read_csv(path, skiprows=10, sep='\t', dtype='unicode');
    dataframe_initial = pd.DataFrame(data);

    #Quantity of file's columns
    qty_col = len(dataframe_initial.columns);

    #Create array for column's names
    column_label_list = [
        'Time'
    ];

    for j in range(qty_col - 1):
        column_label_list.append('S'+ str(j+1))

    column_label_list.append('Scenario');

    #Import scenario for damage
    scenario_number = getDamageScenarioLabel(path);

    # scenario_array = np.full((qty_row), scenario_number).tolist();
    dataframe_initial['Scenario'] = scenario_number;

    dataframe_initial.columns = column_label_list;
    return dataframe_initial

def prepareDatabase(path) :
    """
    Take [list] of paths to loop through and return the statistical caracteristics in a dataframe
    """

    column_label = [];
    for k in range(30):
        column_label.append('S'+ str(k+1))

    ## First to concatenated Dataframe
    df_first_scenario = createDatabase(path[0]);
    df_first_scenario = df_first_scenario.astype(float);

    df_first_sensor = getStatisticalCaracteristics(df_first_scenario,column_label[0]);
    df_first_sensor['Sensor'] = column_label[0];
    df_first_sensor['Scenario'] = df_first_scenario['Scenario'][1];

    df_final = df_first_sensor

    for i in range(len(path)):
        df_scenario = createDatabase(path[i]);
        df_scenario = df_scenario.astype(float);

        for j in range(len(column_label)):
            if (i == 0 and j == 0):
                continue;
            
            df_sensor = getStatisticalCaracteristics(df_scenario,column_label[j]);
            df_sensor['Sensor'] = column_label[j];
            df_sensor['Scenario'] = df_scenario['Scenario'][1];

            df_final = pd.concat([df_final,df_sensor], axis=0, ignore_index=True, sort=False);
    
    return df_final
