# SRP

All the codes are being run on GPU using Multithreading.
Check if you have the right version of pytorch installed for GPU compututation.

pip install virtualenv
First Open the project folder in VSCode and then Create virtual environment using below code
virtualenv venv
.\venv\Scripts\activate

Use below code to uninstall existing torch and reinstall (around 2.3 GB file so connect to Wifi)
the right version [Check in this website] (https://pytorch.org/)

pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Also install these packages by running the below line on Terminal
pip install opencv-python virtualenv matplotlib pandas numpy scikit-learn 


Use parameters.json file to configure the runs
The size of labelled set is decided by images_per_class in json file.
If images_per_class=20, then there are 10 classes in mnist, Labelled pool size will be 20*10=200
Unlabelled size will be 60000-200 = 58000
Play around with Parameters.json file by changing values.


To run the code, just Run the evaluate.py file, which will run all the subsequent files

For results, check the Augumentation Results folder (automatically created at the end of the code), 
which has visualizations and average results for different tecnhiques.