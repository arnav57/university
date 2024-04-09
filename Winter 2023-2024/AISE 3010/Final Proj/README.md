# AISE 3010 Final Project
> By Arnav Goyal and Graeme Watt

---

# Files

Description of the files in this repo...
- `hyperparams.json` : used to store training hyper parameters
- `load_data.py` : needed to obtain data from EEG dataset
- `dataset.py` : needed to cast the EEG data into a pytorch DataSet class
- `models.py` : holds the classes (architectures) of models to use
- `root.py` : holds the backbone of the main execution loop, adjust loss func and dataloaders here also the entry point of program

Some things to keep in mind...
- you need a folder named 'checkpoints'
- you need to download the EEG data and place it into a folder called 'data' it should contain the train.txt and test.txt files