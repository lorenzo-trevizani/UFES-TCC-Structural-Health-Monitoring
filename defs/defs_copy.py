import glob
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft, ifft
import numpy as np
import scipy
import seaborn as sns
import re
from tqdm.notebook import tqdm
import math

#IMPORTING FUNCTIONS FOR FEATURE EXTRACTION
import sklearn.linear_model as LinearModel

#IMPORTING FUNCTIONS FOR CLASSIFIERS
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedShuffleSplit, TimeSeriesSplit, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#IMPORTING FUNCTIONS FOR CLASSIFIERS METRICS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

#OPTMIZE PARAMETHERS
import optuna


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

def getStatisticalCaracteristics(original_df, sensor_column_id, qty_group = 1024, typeof = 'time') :
    """
    Take dataframe and Sensor of choice ('S1') and return the calculated statistical characteristics for group of 50 values
    typeof variable is string that needs to be either 'freq', 'time' or 'both'. 'both' is default
    """

    # Sensor column in dataframe 
    timebased_df = original_df[['Time',str(sensor_column_id)]];
    # [index,    'Time',      'S1'  ]
    # [  0,     0.000000,   0.004954]
    # [ ...,       ...,        ...  ]
    # [ 262k,   255.9999,  -0.211089]
    
    # Sensor column in dataframe 
    variable_group = int(qty_group);

    # Columns for freq database
    freq_columns = ['Amplitude Máxima','Amplitude Média','Amplitude Somatória','Amplitude Desvio Padrão','PSD Máximo','PSD Médio','PSD Somatória','PSD Desvio Padrão'];
    # freq_columns = ['s_std','psd_max'];
    # Columns for timebased database
    # time_columns = ['mean', 'median', 'skew', 'amp_max_min', 'kurtosis'];
    time_columns = ['Média', 'Desvio Padrão', 'Valor Mínimo', 'Valor Máximo', 'Mediana', 'Skewness', 'Amplitude', 'Kurtosis'];

    # Creating heading for columns
    if (typeof == 'both'):
        head_columns = freq_columns + time_columns;
    elif (typeof == 'freq'):
        head_columns = freq_columns;
    else:
        head_columns = time_columns;
    
    # Heading for columns
    grouped_values = pd.DataFrame(columns=head_columns);
    
    for j in range(int(math.floor(len(timebased_df)/variable_group))) :

        df_iteration = timebased_df[j*variable_group:(j+1)*variable_group];
        # [index,    'Time',      'S1'  ]
        # [  0,     0.000000,   0.004954]
        # [ ...,       ...,        ...  ]
        # [ 1023,   0.999023,   0.019900]

        if (typeof == 'both' or typeof == 'freq') :
            # Creating variables for freq input
            ##Getting passo from second time - first time
            passo = round((df_iteration['Time'][(j*variable_group)+1] - df_iteration['Time'][j*variable_group]),8)
            f_new, s_new, n_new, psd_new = df2fft(df_iteration,passo);

            ## NÃO MECHER NA ORDEM, DEVEM SER IGUAL A COLUNAS DO grouped_values
            freq_features = [];
            freq_features.append(np.log10(s_new).max());
            freq_features.append(np.log10(s_new).mean());
            freq_features.append(np.log10(s_new).sum());
            freq_features.append(np.log10(s_new).std());
            freq_features.append(np.log10(psd_new).max());
            freq_features.append(np.log10(psd_new).mean());
            freq_features.append(np.log10(psd_new).sum());
            freq_features.append(np.log10(psd_new).std());
            # freq_features.append((f_new*s_new).mean());
            # freq_features.append((f_new*s_new).max());
            # freq_features.append((f_new*s_new).sum());
            # freq_features.append(f_new[np.argmax(s_new)]);
            # freq_features.append(1/(f_new[np.argmax(s_new)]));
        
        if (typeof == 'both' or typeof == 'time') :
            sensor_df = df_iteration[str(sensor_column_id)];

            ## NÃO MECHER NA ORDEM, DEVEM SER IGUAL A COLUNAS DO grouped_values
            time_features = [];
            time_features.append(sensor_df.mean());
            time_features.append(sensor_df.std());
            time_features.append(sensor_df.min());
            time_features.append(sensor_df.max());
            time_features.append(sensor_df.median());
            time_features.append(sensor_df.skew());
            time_features.append(sensor_df.max() - sensor_df.min());
            time_features.append(pd.Series.kurtosis(sensor_df));
    
        # Creating new row
        if (typeof == 'both'):
            new_row = freq_features + time_features;
        elif (typeof == 'freq'):
            new_row = freq_features;
        else:
            new_row = time_features;
        
        # Creating 'array_features' df with new row data
        array_features = pd.DataFrame([new_row], columns=grouped_values.columns);

        # Concat 'array_features' df in grouped_values
        grouped_values = pd.concat([grouped_values, array_features], ignore_index=True);
    
    new_columns = []
    for i in range(len(grouped_values.columns)):
        new_columns.append(grouped_values.columns[i] + '_' + sensor_column_id)
    grouped_values.columns = new_columns

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

def prepareDatabase(path,number_group = 1024, typeof = 'both') :
    """
    Take [list] of paths to loop through and return the statistical caracteristics in a dataframe
    """

    column_label = [];
    for k in range(30):
        column_label.append('S'+ str(k+1))

    ## First to concatenated Dataframe
    df_first_scenario = createDatabase(path[0]);
    df_first_scenario = df_first_scenario.astype(float);

    df_first_sensor = getStatisticalCaracteristics(df_first_scenario,column_label[0],number_group, typeof);

    for i in range(len(column_label)-1):
        df_first_scenario_sensors = getStatisticalCaracteristics(df_first_scenario,column_label[i+1],number_group, typeof);
        df_first_sensor = pd.concat([df_first_sensor,df_first_scenario_sensors],axis=1)

    df_first_sensor['Scenario'] = getDamageScenarioLabel(path[0]);

    df_final = df_first_sensor

    for i in range(len(path)-1):
        df_scenario = createDatabase(path[i+1]);
        df_scenario = df_scenario.astype(float);

        df_scenario_first_sensor = getStatisticalCaracteristics(df_scenario,column_label[0],number_group, typeof);

        for j in range(len(column_label)-1):
            df_sensor = getStatisticalCaracteristics(df_scenario,column_label[j+1],number_group, typeof);
            df_scenario_first_sensor = pd.concat([df_scenario_first_sensor,df_sensor],axis=1)

        df_scenario_first_sensor['Scenario'] = getDamageScenarioLabel(path[i+1]);

        df_final = pd.concat([df_final,df_scenario_first_sensor], axis=0, ignore_index=True, sort=False);
    
    return df_final

def createDatabaseSingleSensor(df_, sensor) :
    """
    Take dataframe from prepareDatabase and a sensor(int) to return the sensor part of the dataframe with damaged or not label
    """
    pd.set_option('mode.chained_assignment', None);

    #Filter dataframe to get only columns related to the sensor
    regex = r'^(Scenario|.*_S' + str(sensor) +')$';
    df_filtered = df_.filter(regex=regex);

    #Transform scenario column into damaged or not column based on the sensor on focus
    transform = lambda x: 1 if x == str(sensor) else 0;
    df_filtered['Scenario'] = df_filtered['Scenario'].apply(transform);
    df_filtered = df_filtered.rename(columns={'Scenario': 'damaged'});
    return df_filtered

def probabilityOfDamage(predition) :
    """
    Take y_pred from rf and return PoD
    """
    j = 0;
    for i in range(len(predition)) :
        if predition[i] == 1 :
            j = j + 1;

    pod = j/len(predition);
    return pod

def prepareModel(df_new_final) :
    """
    Take df_ and return train,test,scaled
    """

    # Start random forest Model
    df_rf = df_new_final;

    # Setting target variable
    y = df_rf['damaged'];

    # Setting other variables
    X = df_rf.drop(columns=['damaged']);
    X = X.astype(float);

    # Segregate database for test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True) 

    # Scaler database
    std_scaler = StandardScaler()
    X_scaled_train = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_scaled_test = pd.DataFrame(std_scaler.transform(X_test),columns=X_train.columns,index=X_test.index)

    return X_scaled_train, X_scaled_test, y_train, y_test 

def createRFModels(df_rf, qty_sensor=30) :
    """
    Take df_rf and return dataframe with rf_models
    """

    rf_models = pd.DataFrame(columns=['Sensor', 'rf_model', 'accuracy']);

    for i in tqdm(range(qty_sensor)) :
        dataframe_sensor = createDatabaseSingleSensor(df_rf,i+1);
        X_scaled_train, X_scaled_test, y_train, y_test = prepareModel(dataframe_sensor);

        ### DEFAULT PARAMETHERS ON FULL DATAFRAME
        rf = RandomForestClassifier(n_estimators=100);
        rf.fit(X_scaled_train,y_train);
        y_pred = rf.predict(X_scaled_test);
        accuracy_ = accuracy_score(y_test, y_pred);

        d = {'Sensor': [i+1], 'rf_model': [rf], 'accuracy': [accuracy_]};
        appended_file = pd.DataFrame(data=d);

        rf_models = pd.concat([rf_models, appended_file],axis=0);
        
    return rf_models

def createScaledDataframeScenario(df_to_scale, analysed_scenario_input: int) :
    """
    Take df with all features from all sensors and sensor intend to return dataframe with scaled df
    """
    
    df_scaled = df_to_scale;
    analysed_scenario = analysed_scenario_input;
    # CREATE DATAFRAME WITH 2 PARTS OF SCENARIO received, 1 PARTS OF SCENARIO 0 AND 1 PARTS OF ALL OTHER SCENARIOS    
    # CREATE DATAFRAME OF SCENARIO 1
    df_damaged = df_scaled.loc[df_scaled['Scenario'] == str(analysed_scenario)];
    n_damaged = len(df_damaged);

    # CREATE DATAFRAME OF SCENARIO 0
    df_undamaged = df_scaled.loc[df_scaled['Scenario'] == '0'];
    df_undamaged = df_undamaged.sample(n=round(n_damaged/2));
    n_undamaged = len(df_undamaged);

    # CREATE DATAFRAME OF ALL SCENARIOS
    df_random_scenarios = df_scaled.loc[df_scaled['Scenario'] != '0'];
    df_random_scenarios = df_random_scenarios.loc[df_random_scenarios['Scenario'] != str(analysed_scenario)];
    df_random_scenarios = df_random_scenarios.sample(n=round(n_undamaged));

    df_ready = pd.concat([df_damaged, df_undamaged],axis=0);
    df_ready = pd.concat([df_ready, df_random_scenarios],axis=0);

    df_ready = createDatabaseSingleSensor(df_ready, analysed_scenario);
    df_ready = df_ready.reset_index().drop(columns=['index']);
    return df_ready

def df2fft(df_fft,passo_):
    '''
    Função para realizar o FFT.
    Retorna a frequência, spectro, número de observaçoes e o psd
    '''
    n_=len(df_fft)

    fft_calculo=np.fft.fft(df_fft)
    freq=np.fft.fftfreq(n_,d=passo_)
    mascara=freq>0

    fft_abs=2.0*np.abs(fft_calculo/n_)
    psd_calculo=2*(np.abs(fft_calculo/n_)**2)

    s_=fft_abs[mascara] # spectro
    f_=freq[mascara]    # frequencia
    psd_=psd_calculo[mascara]
    # s[0]=s[0]/2

    return f_, s_, n_, psd_

def ar_model(x,n_coef):
    N_training = len(x) 
    x_train = np.zeros((int(N_training - n_coef),n_coef))
    y_train = np.zeros((int(N_training)- n_coef,))
    count = 0
    for i in range(n_coef, N_training):
        x_train[count,:] = x[i - n_coef:i]
        y_train[count,] = x[i]
        count = count + 1

    linear_model = LinearModel.Ridge(alpha=1e-6).fit(x_train, y_train)  
    return(linear_model.coef_)

def getRegressionCaracteristics(original_df, sensor_column_id, qty_group=1024, qty_coef=3):
    """
    Take dataframe and Sensor of choice ('S1') and return the calculated statistical characteristics for group of 50 values typeof variable is string that needs to be either 'freq', 'time' or 'both'. 'both' is default
    """

    # Sensor column in dataframe 
    timebased_df = original_df[['Time',str(sensor_column_id)]];
    # [index,    'Time',      'S1'  ]
    # [  0,     0.000000,   0.004954]
    # [ ...,       ...,        ...  ]
    # [ 262k,   255.9999,  -0.211089]
    df_sensor_it = timebased_df.drop(columns=['Time'])

    # Sensor column in dataframe 
    variable_group = int(qty_group);

    #formando head labels
    head_columns = [];
    for i in range(qty_coef):
        coeff = "coeff_" + str(i);
        head_columns.append(coeff);    
    
    # Heading for columns
    grouped_values = pd.DataFrame(columns=head_columns);

    for j in range(int(math.floor(len(df_sensor_it)/variable_group))) :

        df_iteration = df_sensor_it[j*variable_group:(j+1)*variable_group];
        # [index,    'Time',      'S1'  ]
        # [  0,     0.000000,   0.004954]
        # [ ...,       ...,        ...  ]
        # [ 1023,   0.999023,   0.019900]

        df_iteration = df_iteration[str(sensor_column_id)];

        # Creating new row
        new_row = ar_model(list(df_iteration),int(qty_coef));
        
        # Creating 'array_features' df with new row data
        array_features = pd.DataFrame([new_row], columns=grouped_values.columns);

        # Concat 'array_features' df in grouped_values
        grouped_values = pd.concat([grouped_values, array_features], ignore_index=True);
    
    new_columns = [];
    for i in range(len(grouped_values.columns)):
        new_columns.append(grouped_values.columns[i] + '_' + sensor_column_id);
    grouped_values.columns = new_columns;

    return grouped_values;

def prepareRegressionDatabase(path,number_group = 1024, qty_coef=3) :
    """
    Take [list] of paths to loop through and return the statistical caracteristics in a dataframe
    """

    column_label = [];
    for k in range(30):
        column_label.append('S'+ str(k+1))

    ## First to concatenated Dataframe
    df_first_scenario = createDatabase(path[0]);
    df_first_scenario = df_first_scenario.astype(float);

    df_first_sensor = getRegressionCaracteristics(df_first_scenario,column_label[0],qty_group=number_group, qty_coef=qty_coef);

    for i in range(len(column_label)-1):
        df_first_scenario_sensors = getRegressionCaracteristics(df_first_scenario,column_label[i+1],qty_group=number_group, qty_coef=qty_coef);
        df_first_sensor = pd.concat([df_first_sensor,df_first_scenario_sensors],axis=1)

    df_first_sensor['Scenario'] = getDamageScenarioLabel(path[0]);

    df_final = df_first_sensor

    for i in range(len(path)-1):
        df_scenario = createDatabase(path[i+1]);
        df_scenario = df_scenario.astype(float);

        df_scenario_first_sensor = getRegressionCaracteristics(df_scenario,column_label[0],qty_group=number_group, qty_coef=qty_coef);

        for j in range(len(column_label)-1):
            df_sensor = getRegressionCaracteristics(df_scenario,column_label[j+1],qty_group=number_group, qty_coef=qty_coef);
            df_scenario_first_sensor = pd.concat([df_scenario_first_sensor,df_sensor],axis=1)

        df_scenario_first_sensor['Scenario'] = getDamageScenarioLabel(path[i+1]);

        df_final = pd.concat([df_final,df_scenario_first_sensor], axis=0, ignore_index=True, sort=False);
    
    return df_final
