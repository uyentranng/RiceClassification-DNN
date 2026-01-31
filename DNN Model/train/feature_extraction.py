import polars as pl
import numpy as np
from skimage import io, color, filters, measure
from scipy import ndimage as ndi
import ray
import pywt
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.preprocessing import FunctionTransformer

# Morphological and Shape Feature Extraction
class MorphFeatureExtraction:
    def __init__(self, image_path):
        self.img_path = image_path
        self.grayscaled = color.rgb2gray(io.imread(self.img_path))
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
            'imag_path': self.img_path,
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
    def __init__(self, image_path):
        self.img_path = image_path
        self.rgb = io.imread(self.img_path)
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
            color_features['imag_path'] = self.img_path
            color_features[f"{m}_mean"] = n.mean()
            color_features[f"{m}_std"] = n.std()
            color_features[f"{m}_skewness"] = skew(n, axis=None, nan_policy='omit')
            color_features[f"{m}_kurtosis"] = kurtosis(n, axis=None, nan_policy='omit')
            color_features[f"{m}_entropy"] = shannon_entropy(n)
            color_features[f"{m}_wavelet_decomp"] = self.db_coeff_mean(n)
        return color_features


# batch feature extraction
@ray.remote    
def extract_features_batch(batch):
    try:
        results = []
        for image_path, label in batch:
            try:
                ms_extractor = MorphFeatureExtraction(image_path)
                cl_extractor = ColorFeatureExtraction(image_path)
                features = ms_extractor.morph_shape_feats() | cl_extractor.color_feats()
                results.append((features, label))
            except Exception:
                continue
        return results
    except Exception:
        return []

def split_batches(image_paths, labels, batch_size):
    return [
        list(zip(image_paths[i:i + batch_size], labels[i:i + batch_size]))
             for i in range(0, len(image_paths), batch_size)
    ]

def extract_features(image_paths, labels):
    ray.init(ignore_reinit_error=True, num_cpus=8)
    
    # Split into batches
    batch_size = 300 
    batch = split_batches(image_paths, labels, batch_size)
    tasks = [extract_features_batch.remote(b) for b in batch]
    
    all_results = ray.get(tasks)
    # Flatten list of batches
    flat_results = [item for batch in all_results for item in batch]

    X_dicts, Y = zip(*flat_results)
    X = pl.DataFrame(X_dicts)
    X = X.select(pl.exclude('imag_path'))
    Y = np.array(Y).reshape(-1, 1)
    return X, Y

# single entry feature extraction
def feature_extract_func(img):
    ms_extractor = MorphFeatureExtraction(img)
    cl_extractor = ColorFeatureExtraction(img)
    
    features_dict = ms_extractor.morph_shape_feats() | cl_extractor.color_feats()
    X = pl.DataFrame(features_dict)
    return X

feature_extract = FunctionTransformer(feature_extract_func)


