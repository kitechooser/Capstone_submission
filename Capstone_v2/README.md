This project aims to classify images as real or fake for digital forensics using lightweight models like Mesonet and complementary algorithms (CNN, SVM, Random Forest). It processes 256x256 RGB images with a pipeline including data augmentation (rescaling, rotations, flips) and balanced training via class weights. Key metrics include validation accuracy of 77%. Applications span detecting digital forgeries in journalism, law enforcement, and image authentication. Ethical considerations include potential dataset bias and privacy concerns due to lack of explicit consent. Future improvements target optimizing hyperparameters and combining models to streamline workflow and reduce computational costs.


This code is known working with Python 3.10. This was used so the Apple GPU could be utilised.

There is a notebook in ./src 'training-notebook.ipynb' that shows aspects of this code running with output
There is example output in ./best_models and ./best_models_meso_128x128_run

The data files needed are in the image_data folder. If you prefer to download them they can be found at: https://www.kaggle.com/code/alnomanabdullah/mesonet-with-real-and-fake-images-dataset-1/input

The file 'Model card and datasheet.pdf' holds the datasheet and model card information

NB. although the code runs from the notebook, it is better to run it from the command line with main.py
