## Import modules
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog
from sklearn.svm import LinearSVC
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
import random
import shutil
import pickle
import argparse
import os
import numpy as np
import config as cfg

def data_creation(folder, annot_file, img_folder, benign_base_dir, neg_ratio = 3):
    ## clean folder if destination folder exists
    if os.path.isdir(img_folder):
        shutil.rmtree(img_folder)
        os.mkdir(img_folder)
    else:
        os.mkdir(img_folder)

    for brand in os.listdir(folder):
        if not os.path.isdir(os.path.join(img_folder, brand)): 
            os.mkdir(os.path.join(img_folder, brand)) # create directory for the brand

        if os.path.isdir(os.path.join(folder, brand)):
            for subsite in os.listdir(os.path.join(folder, brand)):
                if os.path.isdir(os.path.join(folder, brand, subsite)): # skip other files, only look at directories
                    ## get the ground-truth label from annotation txt
                    f = [x.strip() for x in open(annot_file).readlines()]
                    for line in f:
                        site = ','.join(line.split(',')[:-4])
                        if site != subsite:
                            continue
                        else:
                            min_x, min_y, max_x, max_y = [float(x) for x in line.split(',')[-4:]]
                            gt_box = [min_x, min_y, max_x, max_y]
                            break

                    ## positive example is cropped from phishing
                    shot = Image.open(os.path.join(folder, brand, subsite, 'shot.png'))
                    pos = shot.crop((min_x, min_y, max_x, max_y))

                    ## save and name the positive example
                    ct = 0
                    for file in os.listdir(os.path.join(img_folder, brand)):
                        if file.startswith('pos') and file.endswith('.png'):
                            ct += 1
                    pos.save(os.path.join(img_folder, brand, 'pos%s.png'%str(ct)))

                    ## repeatly sample from benign and get negative samples
                    for _ in range(neg_ratio):
                        ## randomly sample a benign site
                        benign_shot_list = os.listdir(benign_base_dir)
                        while True:
                            try:
                                benign_shot = Image.open(os.path.join(benign_base_dir, benign_shot_list[random.sample(range(len(benign_shot_list)), 1)[0]], 'shot.png'))
                                break
                            except Exception as e:
                                print(e)
                                continue

                        ## crop and save (negative sample is of the same size as positive for simplicity)
                        ct = 0
                        for file in os.listdir(os.path.join(img_folder, brand)):
                            if file.startswith('neg') and file.endswith('.png'):
                                ct += 1
                        offset_x = random.sample(range(int(0), int(shot.size[0]-max_x)), 1)[0]
                        offset_y = random.sample(range(int(0), int(shot.size[1]-max_y)), 1)[0]
                        neg = benign_shot.crop((min_x + offset_x, min_y + offset_y, max_x + offset_x, max_y + offset_y))
                        neg.save(os.path.join(img_folder, brand, 'neg%s.png'%str(ct)))

        #             plt.imshow(neg)
        #             plt.show()


    ## Split the images into Positive and Negative
    for brand in os.listdir(os.path.join(img_folder)):
        if not os.path.isdir(os.path.join(img_folder, brand, 'Positive')):
            os.mkdir(os.path.join(img_folder, brand, 'Positive'))
        if not os.path.isdir(os.path.join(img_folder, brand, 'Negative')):
            os.mkdir(os.path.join(img_folder, brand, 'Negative'))

        for file in os.listdir(os.path.join(img_folder, brand)):
            if file.startswith('pos') and file.endswith('.png'):
                shutil.move(os.path.join(img_folder, brand, file), os.path.join(img_folder, brand, 'Positive', file))
            elif file.startswith('neg') and file.endswith('.png'):
                shutil.move(os.path.join(img_folder, brand, file), os.path.join(img_folder, brand, 'Negative', file))
  

 
def train(img_folder, logo_size, orientations, pixels_per_cell, cells_per_block):
    
    for brand in os.listdir(img_folder):
        # read the image files:
        pos_im_path = os.path.join(img_folder, brand, 'Positive')
        neg_im_path = os.path.join(img_folder, brand, 'Negative')

        pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
        neg_im_listing = os.listdir(neg_im_path)
        num_pos_samples = size(pos_im_listing) # simply states the total no. of images
        num_neg_samples = size(neg_im_listing)
        print("-"*20 + "Number of positive samples:" + "-"*20, num_pos_samples) # prints the number value of the no.of samples in positive dataset
        print("-"*20 + "Number of negative samples:" + "-"*20, num_neg_samples)


        data= []
        labels = []

        # compute HOG features and label them:
        #this loop enables reading the files in the pos_im_listing variable one by one
        for file in pos_im_listing: 
            img = Image.open(os.path.join(pos_im_path, file)) # open the file
            img = img.resize((logo_size[0], logo_size[1])).convert('RGB')
            # calculate HOG for positive features
            fd = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                     cells_per_block=(cells_per_block, cells_per_block), block_norm='L2', feature_vector=True, multichannel=True)# fd= feature descriptor
            # fd = fd.reshape(-1, 1)
            data.append(fd)
            labels.append(1)  ## label as positive (logo)

        # Same for the negative images
        for file in neg_im_listing:
            img= Image.open(os.path.join(neg_im_path, file))
            img = img.resize((logo_size[0], logo_size[1])).convert('RGB')
            # Now we calculate the HOG for negative features
            fd = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                     cells_per_block=(cells_per_block, cells_per_block), block_norm='L2', feature_vector=True, multichannel=True)
            # fd = fd.reshape(-1, 1)
            data.append(fd)
            labels.append(0)  ## label as positive (non-logo)

        #%% Train the linear SVM
        print(" Training Linear SVM classifier...")
        model = LinearSVC()
        model.fit(np.array(data), np.array(labels))
        #%% Evaluate the classifier
        print(" Evaluating classifier on training data ...")
        predictions = model.predict(np.array(data))
        print(brand)
        print(classification_report(np.array(labels), predictions))
        print(confusion_matrix(np.array(labels), predictions))

        with open(os.path.join(img_folder, brand, 'svm_model.pkl'), 'wb') as handle:
            pickle.dump(model, handle)
                
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_folder", help='Training brands\' screenshots to create positive samples', default= 'benchmark/Train_5brand')
    parser.add_argument('-a', "--annot_file", help='Annotation for training brands', default='benchmark/phish1000_coord.txt' )
    parser.add_argument('-o', '--output_folder', help='Created training folder', default='benchmark/SVM_imageset')
    parser.add_argument('-b', '--benign_folder', help='Benign screenshots to create negative samples', default='benchmark/benign_sample_15k')
    parser.add_argument('-r', '--negpos_ratio', type=int, default=3)
    args = parser.parse_args()
    
    data_creation(args.data_folder, args.annot_file, args.output_folder, args.benign_folder, args.negpos_ratio)
    
    train(args.output_folder, cfg.logo_size, cfg.hog_param['orientation'],
          cfg.hog_param['pixel_per_cell'], cfg.hog_param['cell_per_block'])
    



