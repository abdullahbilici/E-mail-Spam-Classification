# E-mail-Spam-Classification
## YZV 311E Term Project
### Group no: 8

### Members:
* Abdullah Bilici, 150200330
* Bora Boyacıoğlu, 150200310

### Dataset:
https://www.kaggle.com/code/harshsinha1234/email-spam-classification-nlp/input

### Description:
This project aims to create a model for classifying E-Mails as spam or ham (not spam). We tried a couple of different models, and evaluated each of them. Then, used hyperparameter tuning to get the best results. Lastly, we used BERT, which is a special model we tried to get even better results.

### How-To Run:
After installing the necessary libraries, you need to run the Preprocess.ipynb notebook until the end. There will be some data archives created. Next, you will run the model.ipynb file for the generic models, and bert.ipynb for the BERT model.

### Necessary Libraries:
In the requirements.txt file, the libraries used in this project are listed and ready to be installed using "pip install -r requirements.txt". They are:

* transformers==4.35.2
* pandas==2.0.3
* torch==2.1.1
* numpy==1.25.2
* scikit-learn==1.3.2
* spacy==3.7.2
