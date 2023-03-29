import argparse
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str)
args = parser.parse_args()

###### Arguments ###### 
#features: 'AI_TASK', 
#          'MODEL', 'VERSION', 'INPUT_TILE', '#PARAMS', 'STORAGE', 'GFLOP', 'PRECISION', 
#          'PLATFORM', 'ENGINE', 'PROCESSOR'
#       
# Targets: 'INFERENCE_TIME', 'POWER_CONSUMPTION',
#          'CPU_MEM','CPU_MEM_PEAK', 'CPU_LOAD',
#          'GPU_MEM','GPU_MEM_PEAK', 'GPU_LOAD'
#          'QUALITY'

seeds = [0,5,10]
target = args.target

#####################################################################################
# Data Preprocessing
#####################################################################################
###### Load data #####
raw_data = pd.read_csv("20220702153653-benchmark-results.csv")

# The dataset contains observations with missing values ('/') in the columns describing the inference,
# corresponding to the cases in which the DNN fails. Filter out such infeasible configurations:
raw_data = raw_data[raw_data["LATENCY"] != '/']

#####################################################################################
# Feature Design
#####################################################################################
## Features
# Categorical features - One-Hot enconding
onehot_features = ["AI_TASK", "MODEL", "VERSION", "PRECISION", "PLATFORM", "ENGINE", "PROCESSOR"]
onehot_data = pd.get_dummies(raw_data, columns = [col for col in raw_data.columns if col in onehot_features], prefix = "F")
raw_data = pd.concat([raw_data, onehot_data[[col for col in onehot_data.columns if col.startswith("F_")]]], axis = 1)

# Numerical features
if "INPUT_TILE" in raw_data.columns:
    raw_data["F_INPUT_TILE"] = raw_data["INPUT_TILE"].apply(lambda x: x.partition('x'))
    raw_data["F_INPUT_TILE"] = raw_data["F_INPUT_TILE"].apply(lambda x: int(x[0])*int(x[2])).astype(float)
for col in ["#PARAMS", "STORAGE", "GFLOP"]:
    if col in raw_data.columns:
        raw_data["F_" + col] = raw_data[col].astype(float)

# Selecting features
features = [col for col in raw_data.columns if col.startswith("F_")]
X = raw_data[features]

# Normalization:
X = (X - X.min())/(X.max() - X.min())

## Target
# Transform the interested columns into float: 
for col in ['INFERENCE_TIME', 'POWER_CONSUMPTION', 
        'CPU_MEM','CPU_MEM_PEAK', 
        'CPU_LOAD', 'GPU_MEM',
        'GPU_MEM_PEAK', 'GPU_LOAD']:
    raw_data[col] = raw_data[col].astype(float)

# Selecting target
y = raw_data[target]

#####################################################################################
# Training & Testing 
#####################################################################################
# Data Split
idx_train, idx_test = [], []
X_train, X_test = pd.DataFrame(), pd.DataFrame() 
y_train, y_test = pd.Series(dtype=float), pd.Series(dtype=float)
for task in raw_data["AI_TASK"].unique():
    ind = raw_data["AI_TASK"] == task
    F = X[ind]
    g = y[ind]
    #Split
    F_train, F_test, g_train, g_test = train_test_split(F, g, test_size = 0.2, random_state = 10) 
    idx_train = idx_train + F_train.index.tolist()
    idx_test = idx_test + F_test.index.tolist()
    X_train = pd.concat([X_train, F_train], axis = 0, ignore_index = True)
    X_test = pd.concat([X_test, F_test], axis = 0, ignore_index = True)
    y_train = pd.concat([y_train, g_train], axis = 0, ignore_index = True)
    y_test = pd.concat([y_test, g_test], axis = 0, ignore_index = True)

# Define models and metrics
performance = {"Model": [], "Seed" : [], "mse_test": [], "mae_test": [], "mape_test": []} 
predictions = {"True" : y, "Set" : ["Train" if i in idx_train else "Test" for i in raw_data.index]}

# Linear & Tree-based Models
for seed in seeds:
    models_def = {
                "LR" : LinearRegression(), 
                "SVR" : LinearSVR(),
                "RIDGE" : Ridge(),
                "DT" : DecisionTreeRegressor(random_state = seed),
                "DT5" : DecisionTreeRegressor(random_state = seed, max_depth = 5),
                "DT10" : DecisionTreeRegressor(random_state = seed, max_depth = 10),
                "DT15" : DecisionTreeRegressor(random_state = seed, max_depth = 15),
                "DT20" : DecisionTreeRegressor(random_state = seed, max_depth = 20),
                "DT25" : DecisionTreeRegressor(random_state = seed, max_depth = 25),
                "DT30" : DecisionTreeRegressor(random_state = seed, max_depth = 30),
                "RF10" : RandomForestRegressor(random_state = seed, n_estimators = 10),
                "RF25" : RandomForestRegressor(random_state = seed, n_estimators = 25),
                "RF50" : RandomForestRegressor(random_state = seed, n_estimators = 50),
                "RF100" : RandomForestRegressor(random_state = seed, n_estimators = 100),
                "RF150" : RandomForestRegressor(random_state = seed, n_estimators = 150),
                "RF200" : RandomForestRegressor(random_state = seed, n_estimators = 200),
                "RF300" : RandomForestRegressor(random_state = seed, n_estimators = 300)
                }

    for model in models_def:
        # Train
        model_fit = models_def[model].fit(X_train, y_train)
        model_test = model_fit.predict(X_test)
        # Test
        performance["Seed"].append(seed)
        performance["Model"].append(model)
        performance["mse_test"].append(mean_squared_error(y_pred = model_test, y_true = y_test))
        performance["mae_test"].append(mean_absolute_error(y_pred = model_test, y_true = y_test))
        performance["mape_test"].append(mean_absolute_percentage_error(y_pred = model_test, y_true = y_test))
        predictions[model + "_" + str(seed)] = model_fit.predict(X)
        raw_data[model + "_" + str(seed)] = model_fit.predict(X)

# Neural Networks
def NeuralNetwork(hidden_layer_dim = (10, 5), activation = "relu"):
    NN = Sequential()
    NN.add(Dense(hidden_layer_dim[0], input_shape=(len(features),), activation=activation))
    for n in hidden_layer_dim[1:]: 
        NN.add(Dense(n, activation = activation))
    NN.add(Dense(1))
    return NN

optimizer = "Adam"
loss = "mean_absolute_percentage_error"
metrics = ["mean_absolute_percentage_error"] 
epochs = 50
batch_size = 64

models_def = {                   
        "NN.5" : (5,),
        "NN.10" : (10,),
        "NN.20" : (20,),
        "NN.5.5" : (5, 5),
        "NN.10.5" : (10, 5),
        "NN.10.10" : (10, 10),
        "NN.20.10" : (20, 10),
        "NN.20.10.5" : (20, 10, 5),
        "NN.10.10.5.5" : (10, 10, 5, 5),
        "NN.20.10.10.5" : (20, 10, 10, 5),
        "NN.20.20.20.20" : (20, 20, 20, 20)
}

for seed in seeds:
    tf.keras.utils.set_random_seed(seed)
    for model in models_def:
        # Train
        model_fit = NeuralNetwork(models_def[model])
        model_fit.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        model_fit.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        model_test = model_fit.predict(X_test)
        # Test
        performance["Seed"].append(seed)
        performance["Model"].append(model)
        performance["mse_test"].append(mean_squared_error(y_pred = model_test, y_true = y_test))
        performance["mae_test"].append(mean_absolute_error(y_pred = model_test, y_true = y_test))
        performance["mape_test"].append(mean_absolute_percentage_error(y_pred = model_test, y_true = y_test))
        predictions[model + "_" + str(seed)] = model_fit.predict(X).reshape(len(X))
        raw_data[model + "_" + str(seed)] = model_fit.predict(X).reshape(len(X))

# Parse results
performance = pd.DataFrame(performance)
performance = performance.groupby("Model").mean()
performance = performance.drop("Seed", axis = 1)
performance['Model'] = performance.index
print(f'Target: {target}')
print(performance)
predictions = pd.DataFrame(predictions)

#####################################################################################
# Save Results
#####################################################################################
performance.to_csv(f"results/general-evaluation/performance_{target}.csv", index = False)
predictions.to_csv(f"results/general-evaluation/predictions_{target}.csv", index = False)
