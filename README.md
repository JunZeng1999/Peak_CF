# Peak_CF

* Peak_CF is an automatic method based on convolutional neural network (CNN) for image classification and Faster R-CNN for peak location/ classification in untargeted LC-HRMS data.

![1-s2 0-S0003267022007607-ga1_lrg](https://user-images.githubusercontent.com/109707707/180228801-f7531403-c59b-417f-a941-c8f919913054.jpg)

# Cite

* Image classification combined with faster R–CNN for the peak detection of complex components and their metabolites in untargeted LC-HRMS data
* Link: https://doi.org/10.1016/j.aca.2022.340189

# Requirements for Peak_CF operation
* Python, version 3.7 or greater
* Pytorch 1.9.0
* Pycocotools (Windows: pip install pycocotools-windows)
* Windows 10
* Install additional libraries, listed in the requirements.txt

# Usage
* Create three folders for the results of the operations (your path/photo, peak, result)
* Copy the data file (.mzML) to the current directory
* Download the model (link：https://pan.baidu.com/s/1h4B7ZrkiF-twbPTDnf8LgQ?pwd=peak), then unzip it into the save_weights folder.
* Open Peak_CF.py(Modify parameters, Loading the trained model weights, Modify the path where the results are saved)
* Open draw_box_utils.py(Modify parameters, Modify the path to save the result)
* Run Peak_CF.py
* The more detailed instruction on how to use Peak_CF is available via the link(https://pan.baidu.com/s/1VbuS4V3thF6xJnFH9DRu_Q?pwd=peak).
