import tensorflow as tf
import numpy as np
import ray
from ray.train import Checkpoint
import pathlib
import os

dir = pathlib.Path('/Rice_Image_Dataset')

# Load scheckpoint
chkpt = Checkpoint.from_directory(dir / 'checkpoint_000009')

# prepare dataset
test_dataset = ray.data.from_items([
    {"x_test": x, "y_test": y} for x, y in zip(x_test, y_test)
])

# Batch Prediction
def predict_batch(batch: Dict[str, np.ndarray]) -> Dict[str, list]:
    checkpoint = chkpt 
    with checkpoint.as_directory() as checkpoint_dir:
        model = tf.keras.models.load_model(
                os.path.join(checkpoint_dir, 'model.keras')
        )
    y_pred = model(batch['x_test']).numpy()
    pred = np.argmax(y_pred, axis=1)
    pred_encoded = np.eye(num_classes)[pred]
    pred_y = l_encode.inverse_transform(pred_encoded)

    results = []
    if 'y_test' in batch:
        y_true = l_encode.inverse_transform(batch['y_test'])
    else:
        y_true = [None] * len(pred_y)

    for i in range(len(pred_y)):
        results.append({
            'y_pred': pred_y[i],
            'y_true': y_true[i]
        })

    return {"results": results}

def predict_batch_func(x):
    pred = x.map_batches(predict_batch, batch_size=64, concurrency=8)
    prediction_results = []

    for row in pred.iter_rows():
        prediction_results.append(row["results"])
    return prediction_results


# Single Prediction
def predict_func(x, y_true):
    checkpoint = chkpt 
    with checkpoint.as_directory() as checkpoint_dir:
        model = tf.keras.models.load_model(
                os.path.join(checkpoint_dir, 'model.keras')
        )
    x = np.expand_dims(x, axis=0)
    y_pred = model(x).numpy()
    
    result = {}
    for i in range(num_classes):
        pred_encoded = np.eye(num_classes)[i]
        pred_encoded = np.expand_dims(pred_encoded, axis=0)
        key = l_encode.inverse_transform(pred_encoded).item()
        value = y_pred[0][i]
        result[key] = value
          
    y_true = np.expand_dims(y_true, axis=0)
    result['y_true'] = l_encode.inverse_transform(y_true).item()
    return result