# ParaspinalMuscleSeg

This repo contains the code for [Joint Paraspinal Muscle Segmentation and Inter-rater Labeling Variability Prediction with Multi-task TransUNet](https://link.springer.com/chapter/10.1007/978-3-031-16749-2_12). 
Our model is a multi-task TransUNet that provides segmentation masks for paraspinal muscles (left and right erector spinae and multifidus) at the L3-L4,L4-L5, L5-S1, and S1 spinal levels and predicts the pixel-wise variance map of the rater annotations. The tasks in our model have shared convolutional layers and task-specific transformers and decoders.
![Figure-1](https://github.com/HealthX-Lab/ParaspinalMuscleSeg/assets/48385483/5673ba2d-e87a-4547-847c-e98671081340)
## Data Preparation
The network model was trained by using data from the European research consortium project, Genodisc, on commonly diagnosed lumbar pathologies (physiol.ox.ac.uk/genodisc). We are not able to share the data due to the data sharing agreement, but the functions for calculating the variance maps and the majority vote masks are provided in utils.py. For more details about the data, please read the information in ["./data/README.md"](data/README.md).
## Training the Model
Similar to the [original training process of TransUNet](https://github.com/Beckschen/TransUNet/tree/main), we use command-line arguments to set the training parameters. A full list of the parameters can be found in the train script (train.py). If you want to use the default value for all of the parameters, you can simply run
```bash
python train.py
```
## Test
We have included a trained version of our model in ["./trained_models/model_250epochs_0.0033base_lr/epoch_250.pth"](trained_models/model_250epochs_0.0033base_lr). A test notebook is also provided (test.ipynb), along with the code for loading the pre-trained model and calculating the model accuracy for each of the tasks (segmentation and variance map estimation).
## Citations
Roshanzamir, P., et al. (2022). Joint Paraspinal Muscle Segmentation and Inter-rater Labeling Variability Prediction with Multi-task TransUNet. In: Sudre, C.H., et al. Uncertainty for Safe Utilization of Machine Learning in Medical Imaging. UNSURE 2022. Lecture Notes in Computer Science, vol 13563. Springer, Cham. https://doi.org/10.1007/978-3-031-16749-2_12
