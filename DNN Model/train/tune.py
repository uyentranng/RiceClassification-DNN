import tensorflow as tf
from tensorflow import keras as kr 
from tensorflow.keras import layers, optimizers
import numpy as np
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

input_dim = len(X.columns) # input dimension
labels = np.unique(Y)            # labels
num_classes = len(labels)  # number of label classes

# initialize ray
ray.init(ignore_reinit_error=True, num_cpus=8)

# Define search space
search_space = {
    "units_1": tune.choice([32, 64, 128]),
    "units_2": tune.choice([32, 64, 128]),
    "units_3": tune.choice([32, 64, 128]),
    'input_dim': input_dim,
    'learning_rate': tune.choice([0.001, 0.01, 0.1]),
    'batch_size': tune.choice([128, 256, 512]),
    'dropout_rate': 0.2,
    'epochs': 10,
    'validation_split': 0.2
    }

# x_train, y_train into ray object store
x_train_id = ray.put(x_train)
y_train_id = ray.put(y_train)

def objective(search_space):
    # build a hypermodel
    hypermodel = kr.Sequential()
    hypermodel.add(layers.Input(shape=(input_dim,)))
    hypermodel.add(layers.Dense(units=search_space['units_1'], activation='relu'))
    hypermodel.add(layers.Dropout(search_space['dropout_rate']))
    hypermodel.add(layers.Dense(units=search_space['units_2'], activation='relu'))
    hypermodel.add(layers.Dense(units=search_space['units_3'], activation='relu'))
    hypermodel.add(layers.Dense(units=num_classes, activation='softmax'))
    optimizer = optimizers.Adam(search_space['learning_rate'])
    hypermodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
   
    # Retrieve x_train and y_train from object store
    x_train_local = ray.get(x_train_id)
    y_train_local = ray.get(y_train_id)

    
    # fit the model
    history = hypermodel.fit(
                x_train_local,
                y_train_local,
                batch_size=int(search_space.get('batch_size', 64)),
                epochs=int(search_space.get('epochs')),
                validation_split=search_space.get('validation_split'),
                callbacks=[kr.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                verbose=0
                ) 
    
    # extract and report metrics
    tune.report({
        'val_loss': history.history['val_loss'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    })
    

algo = HyperOptSearch()

tuner = tune.Tuner(
    tune.with_resources(objective, resources={"cpu": 8}),
    tune_config=tune.TuneConfig(
        metric="val_accuracy", 
        mode="max",
        search_alg=algo,
        num_samples=98
    ),
    param_space=search_space,
)
results = tuner.fit()

# Print the best result
best_result = results.get_best_result(metric="val_accuracy", mode="max")
best_result_dataframe = best_result.metrics_dataframe