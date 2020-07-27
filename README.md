# SVM-HOG-object-detection
A Supported-vector-machine approach for object detection

- Borrowed from https://github.com/SamPlvs/Object-detection-via-HOG-SVM

# Project Structure
```
|__config.py: 
|        the parameters for HOG, sliding window size, common logo size
|__train.py: 
|        training script
|__test.py
|        testing script
|__utils.py
|        neccessary functions (sliding window generation, prediction)
```
# Instructions
- Clone this repository and install requirements
```
    git clone https://github.com/lindsey98/SVM-HOG-object-detection.git
```

- Install requirements
```
    pip install -r requirements.txt
```
- Train a SVM classifier for logo/non-logo patches
``` 
$ python train.py [options/default]
options:
    -d 
        Training screenshots used to create positive samples
    -a 
        Logo position labels for positive samples
    -o 
        Output directory to save training samples as well as models
    -b 
        Benign screenshots to create negative samples
    -r 
        Ratio of negative samples over positive samples
```

- Test the classifer 
```
$ python test.py [options/default]
    -p 
        path to output file
    -f 
        test data folder which contains screenshots
    -m
        folder with pre-saved svm models
    
```

## Examples 
- Bank of America logo is successfully detected from the screenshot
![Sample output from SVM](img/boa.png)