# TensorFlow CUB200 
CUB200 Example Code

## Download the dataset

You can find the dataset website [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The dataset files are relatively small (about 1.3 GB when untared) and should easily fit on your machine.  

```
$ wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
$ tar -xzf CUB_200_2011.tgz
```

## Create the tfrecord files 

We will use the [tfrecords repo](https://github.com/visipedia/tfrecords) to create the tfrecord files that we can use to train and test the model. You'll need to clone that repo:
```
$ cd ~/code
$ git clone https://github.com/visipedia/tfrecords.git
```

Before we can call the `create()` method in [create_tfrecords.py](https://github.com/visipedia/tfrecords/tree/master/create_tfrecords.py) we will need to format the CUB data. We'll use the [format.py](format.py) script for doing this. Fire up an ipython terminal:
```python
import json

import format as cub_formatter

# Change these paths to match the location of the CUB dataset on your machine 
cub_dataset_dir = "/home/ubuntu/datasets/CUB_200_2011"
cub_image_dir = "/home/ubuntu/datasets/CUB_200_2011/images"

# we need to create a file containing the size of each image in the dataset. 
# you only need to do this once. scipy is required for this method. 
# Alternatively, you can create this file yourself. 
# Each line of the file should have <image_id> <width> <height>
cub_formatter.create_image_sizes_file(cub_dataset_dir, cub_image_dir)

# Now we can create the datasets
train_val, test = cub_formatter.format_dataset(cub_dataset_dir, cub_image_dir)
train, val = cub_formatter.create_validation_split(train_val, fraction_per_class=0.1, shuffle=True)

# We can save off these datasets for convenience
with open('train_dataset.json', 'w') as f:
  json.dump(train, f)
with open('val_dataset.json', 'w') as f:
  json.dump(val, f)
with open('test_dataset.json', 'w') as f:
  json.dump(test, f)
```
We have created three arrays holding train, validation and test data. The CUB-200 dataset does not come with a standard validation set, so we took 10% of the train data and created a validation set. The number of elements in each array should be:
 * Number of train images: 5394
 * Number of validation images: 600
 * Number of test images: 5794

We can now pass these arrays to the `create()` method of the tfrecords repo:
```python

# You might need to add the tfrecords directory to your path
import sys
sys.path.append("/home/ubuntu/code/tfrecords")

from create_tfrecords import create

# Change this path
dataset_dir = "/home/ubuntu/tfrecord_datasets/cub/with_600_val_split/"

train_errors = create(dataset=train, dataset_name="train", output_directory=dataset_dir,
                      num_shards=5, num_threads=1, shuffle=True)

val_errors = create(dataset=val, dataset_name="val", output_directory=dataset_dir,
                    num_shards=1, num_threads=1, shuffle=True)

test_errors = create(dataset=test, dataset_name="test", output_directory=dataset_dir,
                     num_shards=5, num_threads=1, shuffle=True)
```

We now have a dataset directory containing tfrecord files prefixed with either `train`, `val` or `test` that we can use to train and test a model. 

I'll assume that the path to the dataset directory is stored in the `DATASET_DIR` environment variable for the experiments, for example:
```
$ export DATASET_DIR=/media/drive2/tensorflow_datasets/cub/with_600_val_split
```
