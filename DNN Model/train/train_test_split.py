# spilt datasets into to train and test set
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import polars as pl
import numpy as np
import joblib

X = X.select(pl.exclude('imag_path'))
Y = np.array(Y).reshape(-1, 1)

# encoding label variable
l_encode = OneHotEncoder(sparse_output=False)
y = l_encode.fit_transform(Y)

# splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scaling data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# save encoder and scaler
joblib.dump(l_encode, dir / 'data_preprocess/label_encoder.save') 
joblib.dump(scaler, dir / 'data_preprocess/scaler.save') 
# l_encode = joblib.load(dir / 'data_preprocess/label_encoder.save') 
# scaler = joblib.load(dir / 'data_preprocess/scaler.save') 

# build data preprocessing pipeline
preprocess_pipeline = Pipeline([
    ('read_image', read_image),
    ('feature_extract', feature_extract),
    ('scaler', scaler)
])

joblib.dump(preprocess_pipeline, dir / 'data_preprocess/pipeline.save')