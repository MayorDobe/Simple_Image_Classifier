# Simple Image Classifier

This repository contains the my adapted approach to training a simple image classification model based on the popular  Kaggle notebook by Rob Mulla (https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)

## Getting Started

* Clone this repo to your local machine.
```bash
git clone https://github.com/MayorDobe/Simple_Image_Classifier.git
```
Set up enviroment.
* conda
```bash
conda create -n simple_image_classifier python=3.13
conda activate simple_image_classifier
```
* venv
```bash
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/enviroment/bin/activate
```
* Install required packages.
```bash
pip install -r requirements.txt
```

* Generate a Kaggle api key to download our dataset

Authentication https://www.kaggle.com/docs/api

In order to use the Kaggle’s public API, you must first authenticate using an API token. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.

* Once we have our api key in ~/.kaggle/kaggle.json we can pull down our dataset.
```bash
cd ~/Simple_Image_Classifier
mkdir dataset && cd dataset
kaggle datasets download gpiosenka/cards-image-datasetclassification --unzip
```

## Usage.
The project contains two scripts one for training our model with a limited number of parameters that can be changed located in the config dictionary of training_script.py
```python
config = {
    "model": "efficientnet_b0",
    "optimizer": optim.Adam,
    "criterion": nn.CrossEntropyLoss(),
    "batch_size": 32,
    "num_epochs": 6,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
```
To train the model we execute our script. after completion a graph will plot our loss.
```bash
python training_script.py
```

To evaluate our model we run our evauation script. This will give us our models accuracy rating along with a number of set examples showing our models probabilities.
``` bash
python evaluation_script.py
```

## TODO
* Add functionality to allow command line arguments to be passed.
* Create a config.json file for easier manipulation of hyper-params
* Add functionality to allow more than a single model to be trained and saved.
* Add functionality to display better metrics of our model.
* learn
* fail
* succeed
* repeat








