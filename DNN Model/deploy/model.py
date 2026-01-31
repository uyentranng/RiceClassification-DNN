import numpy as np
import polars as pl
from skimage import io, color, filters, measure
from scipy import ndimage as ndi
from pathlib import Path
from ray.train import Checkpoint
import pywt
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import tensorflow as tf
from flask import Flask, request, render_template
from pathlib import Path
from werkzeug.utils import secure_filename

dir = Path('/Rice_Image_Dataset')
template_folder = Path('/RiceClassification/DNN Model/deploy/templates')
static_folder = Path('/RiceClassification/DNN Model/deploy/static')
UPLOAD_FOLDER = static_folder / 'uploads'

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# read image path
def read_image_func(img_path):
    img = io.imread(img_path)
    return img

read_image = FunctionTransformer(read_image_func)

# Morphological and Shape Feature Extraction
class MorphFeatureExtraction:
    def __init__(self, img):
        self.img = img
        self.grayscaled = color.rgb2gray(img)
        self.edge_filtered = filters.sobel(self.grayscaled)
        binary_edges = self.edge_filtered > 0.1
        fill = ndi.binary_fill_holes(binary_edges)
        # Label connected components
        self.labeled_image = measure.label(fill)
    
    # define features that are not included in skimage.measure.regionprops
    @staticmethod
    def measure_compactness(perimeter, num_pixel):
        if (num_pixel).all() <= 0:
            raise ValueError("num_pixel must be greater than 0")
        cd = (4 * num_pixel - perimeter) / 2
        cd_min = num_pixel - 1
        cd_max = ((8 * num_pixel) - 4 * (num_pixel ** (1 / 2))) / 2
        return (cd - cd_min) / (cd_max - cd_min)

    @staticmethod
    def measure_roundness(perimeter, area):
        if (area).all() <= 0:
            raise ValueError("Area must be greater than 0")
        return perimeter / (4 * np.pi * area)

    @staticmethod
    def measure_aspect_ratio(min_row, min_col, max_row, max_col):
        if max_col - min_col == 0:
            raise ValueError("The width (max_col - min_col) cannot be zero.")
        return (max_row - min_row) / (max_col - min_col)

    def morph_shape_feats(self):
        props = measure.regionprops(self.labeled_image)
        
        if not props:
            return None  # Skip if no regions are found
        
        # Get the first region's properties
        region = props[0]

        # Dictionary to hold feature data
        morph_shape_features = {
            'area': region.area,
            'area_convex': region.convex_area,
            'axis_major_length': region.major_axis_length,
            'axis_minor_length': region.minor_axis_length,
            'eccentricity': region.eccentricity,
            'extent': region.extent,
            'perimeter': region.perimeter,
            'solidity': region.solidity,
            'equivalent_diameter_area': region.equivalent_diameter,
            'roundness': self.measure_roundness(region.perimeter, region.area),
            'aspect_ratio': self.measure_aspect_ratio(region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3]),
            'compactness': self.measure_compactness(region.perimeter, region.area),
            'shape1': region.major_axis_length / region.area,
            'shape2': region.minor_axis_length / region.area,
            'shape3': region.area / (np.pi * region.minor_axis_length ** 2),
            'shape4': region.area / (np.pi * (region.major_axis_length / 2) * (region.minor_axis_length / 2))
        }
        return morph_shape_features


# Color Feature Extraction
class ColorFeatureExtraction:
    def __init__(self, img):
        self.img = img
        self.rgb = self.img
        self.hsv = color.rgb2hsv(self.rgb)
        self.lab = color.rgb2lab(self.rgb)
        self.ycbcr = color.rgb2ycbcr(self.rgb)
        self.xyz = color.rgb2xyz(self.rgb)
        self.color_space = [self.rgb, self.hsv, self.lab, self.ycbcr, self.xyz]
    
    def split_color_space(self):
        channel = ['rgb_r', 'rgb_g', 'rgb_b', 'hsv_h', 'hsv_s', 'hsv_v', 'lab_l', 'lab_a', 'lab_b',
                   'ycbcr_y', 'ycbcr_cb', 'ycbcr_cr', 'xyz_x', 'xyz_y', 'xyz_z']
        channel_value = []
        for i in self.color_space:
            for j in range(i.shape[2]):
                channel_value.append(i[:, :, j])
        self.channel_values = {ch: val for ch, val in zip(channel, channel_value)}
        return self.channel_values
    
    @staticmethod
    def im_entropy(im):
        edge_filtered = filters.sobel(im)
        binary_edges = edge_filtered > 0.1
        fill = ndi.binary_fill_holes(binary_edges)
        labeled_image = measure.label(fill)
        entr_img = entropy(image=im, footprint=disk(7), mask=labeled_image)
        return entr_img
    
    @staticmethod
    def db_coeff_mean(im):
        coeffs = pywt.dwt2(data=im, wavelet='db4')
        cA, (cH, cV, cD) = coeffs
        coeffs_mean = (cA.mean() + cH.mean() + cV.mean() + cD.mean()) / 4
        return coeffs_mean
    
    def color_feats(self):
        color_features = {}
        color_channel = self.split_color_space()
        for m, n in color_channel.items():
            color_features[f"{m}_mean"] = n.mean()
            color_features[f"{m}_std"] = n.std()
            color_features[f"{m}_skewness"] = skew(n, axis=None, nan_policy='omit')
            color_features[f"{m}_kurtosis"] = kurtosis(n, axis=None, nan_policy='omit')
            color_features[f"{m}_entropy"] = shannon_entropy(n)
            color_features[f"{m}_wavelet_decomp"] = self.db_coeff_mean(n)
        return color_features


# feature extraction
def feature_extract_func(img):
    ms_extractor = MorphFeatureExtraction(img)
    cl_extractor = ColorFeatureExtraction(img)
    features_dict = ms_extractor.morph_shape_feats() | cl_extractor.color_feats()
    X = pl.DataFrame(features_dict)
    return X

feature_extract = FunctionTransformer(feature_extract_func)


# load prefitted encoder and scaler
l_encode = joblib.load(dir / 'data_preprocess/label_encoder.save') 
scaler = joblib.load(dir / 'data_preprocess/scaler.save')


# build preprocessing pipeline
preprocess_pipeline = Pipeline([
    ('read_image', read_image),
    ('feature_extract', feature_extract),
    ('scaler', scaler)
])

# load model 
checkpoint = Checkpoint.from_directory(
    dir / 'TensorflowTrainer_2025-06-10_17-39-34/TensorflowTrainer_1bfae_00000_0_2025-06-10_17-39-34/checkpoint_000009')

with checkpoint.as_directory() as checkpoint_dir:
        model = tf.keras.models.load_model(
                os.path.join(checkpoint_dir, 'model.keras')
        )
        
def predict_func(X):
    y_pred = model(X).numpy()
    
    num_classes = 5
    pred = np.argmax(y_pred, axis=1)
    pred_encoded = np.eye(num_classes)[pred]
    pred_class = l_encode.inverse_transform(pred_encoded)[0][0]
    pred_prob = y_pred[0][pred][0]
    return pred_class, round(pred_prob * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['img_file']
        filename = secure_filename(img.filename)
        img_path =  os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(img_path)
        processed_img = preprocess_pipeline.transform(str(img_path))
        pred_class, pred_prob = predict_func(processed_img)
        return render_template('index.html', pred_class=pred_class, pred_prob=pred_prob, filename=filename)
   
    return render_template('index_html')

if __name__ == "__main__":
    app.run(debug=True)