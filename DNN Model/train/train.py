import tensorflow as tf
from tensorflow import keras as kr 
from tensorflow.keras import layers, optimizers
import tempfile
import os
import json
import ray
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
from ray.train import Checkpoint, get_checkpoint, get_dataset_shard, report


class MyDNNModel:
    """get the model with the highest validation accuracy to train"""
    def best_model(self, hypermodel, units_tuner):
        hypermodel = hypermodel
        best_hp = units_tuner.get_best_hyperparameters(1)
        model = hypermodel.build(best_hp)
        return model

    def fit(self, hp, model, x, y, validation_split=None, **kwargs):
        return model.fit(
            x,
            y,
            batch_size=hp.Choice('batch_size', [64, 256, 512]),
            validation_split=validation_split,
            **kwargs
        ) 
    
# Checkpoint Configuration
from ray.train import RunConfig, CheckpointConfig

# Only keep the best checkpoint and delete the others.
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        # *Best* checkpoint are determined by these params:
        checkpoint_score_attribute='accuracy',
        checkpoint_score_order='max',
    ),
    storage_path=dir,
)

# create a ray dataset
def create_ray_dataset(x, y):
    dataset = ray.data.from_items([
        {"x_train": x, "y_train": y} for x, y in zip(x, y)
    ])
    return dataset

# initialize ray
ray.init(ignore_reinit_error=True, num_cpus=8)

# get best config results from hyperparameter tuning
best_config = best_result.config

"""build the model with the highest validation accuracy to train"""
def build_best_model(best_config: dict):
    model = kr.Sequential()
    model.add(layers.Input(shape=(input_dim,))) 
    model.add(layers.Dense(units=best_config['units_1'], activation='relu'))
    model.add(layers.Dropout(best_config['dropout_rate']))
    model.add(layers.Dense(units=best_config['units_2'], activation='relu'))
    model.add(layers.Dense(units=best_config['units_3'], activation='relu'))
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    optimizer = optimizers.Adam(best_config['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# create the train function
def train_func(config: dict):
    
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    with strategy.scope():
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                model = kr.models.load_model(
                    os.path.join(checkpoint_dir, 'model.keras')
                )
        else:
            model = build_best_model(config)

    dataset = get_dataset_shard('train_dataset')
    
    results = []
    for epoch in range(config.get('epochs')):
        tf_dataset = dataset.to_tf(
            feature_columns=['x_train'], label_columns='y_train', batch_size=config.get('batch_size')
        )
        history = model.fit(
            tf_dataset,
            callbacks=[kr.callbacks.EarlyStopping(monitor='loss', patience=3)]
        )
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model.save(os.path.join(temp_checkpoint_dir, 'model.keras'))
            extra_json = os.path.join(temp_checkpoint_dir, 'checkpoint.json')
            with open(extra_json, 'w') as f:
                json.dump({'epoch': epoch}, f)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            
            report({
                'loss': history.history['loss'][-1],
                'accuracy': history.history['accuracy'][-1]}, checkpoint=checkpoint)
        results.append(history.history)
    return results


scaling_config = ScalingConfig(num_workers=6)
trainer = TensorflowTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=best_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={'train_dataset': create_ray_dataset(x_train, y_train)}
)

training_result = trainer.fit()   
    