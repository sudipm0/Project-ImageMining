# Efficient Image Mining & Classification Techniques#

**Project Goal**:
  The project aims to study on below topics/problems:

    1. If a new image is added to a folder, is it causing anomaly in the existing dataset? e.g. training dataset is already labeled
    2. IF a model is trained with a set of images can it detect contamination/anomaly in another folder?
    3. Cluster images by their feature set.
   
**Run as:**:

  `python driver.py`

  `[arguments]   [default]   [options]`

   `--backbone    resnet  resnet/vgg`

   `--dataset     food5k  food5k/PascalVOC`

   `--task        anomaly anomaly/cluster`

   `--subtask     novelty novelty/outlier`

**Pre-Requisites**:

- Python 3.6
- Tensorflow
- Keras
- sklearn

`Note: It is assumed that datasets used in this projects are downloaded into "images" folder. Will update the code
to get the dataset online and prepare the folder structures.`

**Methodology**:

