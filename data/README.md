## Preparing the train, validation, and test data
For the original model training of this paper, the data for train, validation, and test was provided in separate Pandas dataframes. 
In each dataframe, the rows represented different subjects and each row contained all the information (images, majority vote masks, and variance maps) about that specific subject. Below you can see an example of the rater disagreements for one mask and the calculated majority vote.
![Figure 1](https://github.com/HealthX-Lab/ParaspinalMuscleSeg/assets/48385483/58dd35e3-674f-49ee-8711-b001bab85b83)
During the train and test processes, the data is loaded from the corresponding dataframes (stored as .pkl files) and re-sorted as list of dictionaries of {'mri': , 'maj': , 'var': }. The functions for loading the data for train and test are provided in the utils module (utils.py file).
