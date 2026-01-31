# Classification of Rice Varieties with Deep Neural Network
This project aims reproduce the work of Koklu et al. (2021), "Classification of rice varieties with deep learning methods". A Deep Neural Network (DNN) is employed to classify five rice varieties based on image data. The feature dataset was constructed following the methodology proposed by Koklu et al. (2021) in the aforementioned study.

### Dataset
Data source: https://www.muratkoklu.com/datasets/

The dataset consists of 75000 images of five rice varieties: Arborio, Basmati, Ipsala, Jasmine and Karacadag, 15000 from each varieties. A second feature dataset are then constructed, which includes 12 morphological, 4 shape factors and 90 color features (Feature Engineering).

List of morphological features:
|   |                    |   |                     |    |                     |   
|---| ------------------ |---| ------------------- |--- | ------------------- | 
| 1 | Area               | 5 | Eccentricity        |  9 | Eccentricity        |
| 2 | Perimeter          | 6 | Equivalent Diameter | 10 | Eccentricity        |
| 3 | Major Axis Length  | 7 | Solidity            | 11 | Eccentricity        |
| 4 | Minor Axis Length  | 8 | Convex Area         | 12 | Eccentricity        |

List of shape factors:
|   |                    |                                                              |   
|---| ------------------ |------------------------------------------------------------- | 
| 1 | shape factor 1     | Major Axis Length / Area                                     |
| 2 | shape factor 2     | Minor Axis Length / Area                                     |
| 3 | shape factor 3     | Area / (pi * (Major Axis Length / 2) ** 2)                   |
| 4 | shape factor 4     | Area / (pi * (Major Axis Length / 2) * (Minor Axis Length 2) |

List of color features
| Color Space |     Mean    | Standard Deviation | Skewness        | Kurtosis        | Entropy        | Wavelet Decomposition|
|-------------|-------------|--------------------| --------------- |---------------- | -------------- | --------------|
|RGB          |Mean_RGB_R   |     StdDev_RGB_R   |Skewness_RGB_R   |Kurtosis_RGB_R   |Entropy_RGB_R   |Daub4_RGB_R    |
|             |Mean_RGB_G   |     StdDev_RGB_G   |Skewness_RGB_G   |Kurtosis_RGB_G   |Entropy_RGB_G   |Daub4_RGB_G    |
|             |Mean_RGB_B   |     StdDev_RGB_B   |Skewness_RGB_B   |Kurtosis_RGB_B   |Entropy_RGB_B   |Daub4_RGB_B    |
|HSV          |Mean_HSV_H   |     StdDev_HSV_H   |Skewness_HSV_H   |Kurtosis_HSV_H   |Entropy_HSV_H   |Daub4_HSV_H    |
|             |Mean_HSV_S   |     StdDev_HSV_S   |Skewness_HSV_S   |Kurtosis_HSV_S   |Entropy_HSV_S   |Daub4_HSV_S    |
|             |Mean_HSV_V   |     StdDev_HSV_V   |Skewness_HSV_V   |Kurtosis_HSV_V   |Entropy_HSV_V   |Daub4_HSV_V    |
|L*a*b*       |Mean_LAB_L   |     StdDev_LAB_L   |Skewness_LAB_L   |Kurtosis_LAB_L   |Entropy_LAB_L   |Daub4_LAB_L    |
|             |Mean_LAB_A   |     StdDev_LAB_A   |Skewness_LAB_A   |Kurtosis_LAB_A   |Entropy_LAB_A   |Daub4_LAB_A    |
|             |Mean_LAB_B   |     StdDev_LAB_B   |Skewness_LAB_B   |Kurtosis_LAB_B   |Entropy_LAB_B   |Daub4_LAB_B    | 
|YCbCr        |Mean_YCbCr_Y |    StdDev_YCbCr_Y  |Skewness_YCbCr_Y |Kurtosis_YCbCr_Y |Entropy_YCbCr_Y |Daub4_YCbCr_Y  |
|             |Mean_YCbCr_Cb|    StdDev_YCbCr_Cb |Skewness_YCbCr_Cb|Kurtosis_YCbCr_Cb|Entropy_YCbCr_Cb|Daub4_YCbCr_Cb |
|             |Mean_YCbCr_Cr|    StdDev_YCbCr_Cr |Skewness_YCbCr_Cr|Kurtosis_YCbCr_Cr|Entropy_YCbCr_Cr|Daub4_YCbCr_Cr |
|XYZ          |Mean_XYZ_X   |     StdDev_XYZ_X   |Skewness_XYZ_X   |Kurtosis_XYZ_X   |Entropy_XYZ_X   |Daub4_XYZ_X    |
|             |Mean_XYZ_Y   |     StdDev_XYZ_Y   |Skewness_XYZ_Y   |Kurtosis_XYZ_Y   |Entropy_XYZ_Y   |Daub4_XYZ_Y    |
|             |Mean_XYZ_Z   |     StdDev_XYZ_Z   |Skewness_XYZ_Z   |Kurtosis_XYZ_Z   |Entropy_XYZ_Z   |Daub4_XYZ_Z    |


### Deep Neural Network
The DNN comprises three hidden layers to facilitate the learning process. A dropout method is applied to mitigate overfitting. Hyperparameter optimization is performed using HyperOptSearch, which can take as input a null prior distribution specification and the experimental history of loss function evaluations to suggest promising configurations for the next trials (Bergstra et al., 2013)

### Literature
Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

Bergstra, J., Yamins, D., & Cox, D. (2013). Making a science of model search: hyperparameter optimization in hundreds of dimensions for vision architectures. Digital Access to Scholarship at Harvard (DASH) (Harvard University), 115–123. http://nrs.harvard.edu/urn-3:HUL.InstRepos:12561000