import os
import pickle
import time
from utils import *
import argparse


def result_gen(write_path, folder, model_dir):
    with open(write_path, 'w') as f:
        f.write('Folder name')
        f.write('\t')
        f.write('Prediction')
        f.write('\t')
        f.write('Prob')
        f.write('\t')
        f.write('Runtime')
        f.write('\n')

    for brand in os.listdir(folder):
        ## open screenshot
        shot_path = folder + '/' + brand + '/shot.png'

        ## generate sliding windows
        start_time = time.time()

        pred = ''
        max_prob = 0
        for target in os.listdir(model_dir):
            with open(model_dir + '/' + target + '/svm_model.pkl', 'rb') as handle:
                model = pickle.load(handle)
            sc, pick, img = pred_logosense(model=model,
                                           shot_path=shot_path)
            if len(sc):
                if max(sc) >= max_prob:
                    max_prob = max(sc)
                    pred = target

        runtime = time.time() - start_time

        with open(write_path, 'a+') as f:
            f.write(folder + '/' + brand)
            f.write('\t')
            f.write(pred)
            f.write('\t')
            f.write(str(max_prob))
            f.write('\t')
            f.write(str(runtime))
            f.write('\n')

        print("-"*20 + "True brand: " + brand.split('+')[0] + "-"*20)
        print("-"*20 + "Predicted brand: " + pred + "-"*20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--output_path", help='Where you save the result txt file',
                         default= 'D:/ruofan/git_space/phishpedia/benchmark/logosense_test.txt')

    parser.add_argument('-f', "--folder", help='Folder you want to test',
                        default='D:/ruofan/git_space/phishpedia/benchmark/test15k_wo_localcontent/benign_sample_15k' )

    parser.add_argument('-mdir', '--model_dir', help='Models for 5 brands', default='D:/ruofan/git_space/phishpedia/benchmark/SVM_imageset')
    args = parser.parse_args()

    result_gen(args.output_path, args.folder, args.model_dir)
