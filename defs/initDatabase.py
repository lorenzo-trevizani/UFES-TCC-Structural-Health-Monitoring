import pandas as pd
import numpy as np
import getDamageScenario

def initDatabase(path) :
    """
    Take list of paths to loop through and return the dataframe unified and merge with damage scenario
    """
    ## Get column names from database
    data = pd.read_csv(path[0], skiprows=10, sep='\t', header=None, dtype='unicode');
    dataframe_initial = pd.DataFrame(data);

    #Quantity of file's columns
    qty_col = len(data.columns);

    #Create array for column's names
    column_label_list = [];

    for j in range(qty_col):
        column_label = dataframe_initial[j][0];
        column_label_list.append(column_label);

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
        scenario_number = getDamageScenario.getDamageScenario(path[i]);
        scenario_array = np.full((qty_row), scenario_number).tolist();

        #Merge scenario number into dataframe
        df[qty_col] = scenario_array;

        ## Assign df to empty list
        appended_file.append(df);

        #Merge scenario number into dataframe
        dataframe_final = pd.concat(appended_file, axis=0, ignore_index=True, sort=False);
        
    dataframe_final.columns = column_label_list;

    return dataframe_final;
