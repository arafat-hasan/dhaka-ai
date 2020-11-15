import cv2
import json
import numpy as np
import os
import time
import glob
import csv
import sys
import argparse

import  tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append('../EfficientDet/')

from model import efficientdet
from utils import preprocess_image, postprocess_boxes

def get_class_names(class_path):
    with open(class_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return {v:k for v, k in enumerate(content)}


def check_badbox(file, img_height, img_width, x_min, y_min, x_max, y_max):
    flag = False
    if x_max > img_width:
        print("Badbox (x_max > img_width) found in: "+file)
        print("x_max: ", x_max)
        print("img_width: ", img_width)
        flag = True
    if x_max < 0:
        print("Badbox (x_max < 0) found in: "+file)
        flag = True
    if y_max > img_height:
        print("Badbox (y_max > img_height) found in: "+file)
        flag = True
    if y_min < 0:
        print("Badbox (y_min < 0) found in: "+file)
        flag = True
    return flag

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = opt.phi
    weighted_bifpn = False
    model_path = opt.model
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]

    # Dhaka-ai classes
    dhaka_ai_classes = get_class_names(opt.class_names)
    dhaka_ai_num_classes = len(dhaka_ai_classes)

    score_threshold = opt.conf_thres

    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=dhaka_ai_num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)

    with open('submission_files/arafat_efficientdet-result_conf-{}_IOUthr-{}_{}_ac-0.0.csv'.format(opt.conf_thres, opt.iou_thres, time.strftime("%Y-%m-%d_%H-%M-%S")), mode='w') as result_file:
        fieldnames = ['image_id', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']
        result_file_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_file_writer.writerow(fieldnames)
        for image_path in glob.glob(os.path.join(opt.source_dir, "*.jpg")):
            image = cv2.imread(image_path)
            assert image is not None, "Image cat not be read, path: "+image_path

            # BGR -> RGB
            image = image[:, :, ::-1]
            h, w = image.shape[:2]

            image, scale = preprocess_image(image, image_size=image_size)
            # run network
            start = time.time()
            boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
            boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
            print(time.time() - start)
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
            
            selected_indices = tf.image.non_max_suppression(
                boxes, scores, 120, iou_threshold=opt.iou_thres, score_threshold=opt.conf_thres)
            selected_boxes = tf.gather(boxes, selected_indices)
            selected_labels = tf.gather(labels, selected_indices)
            selected_boxes = tf.Session().run(selected_boxes)
            selected_labels = tf.Session().run(selected_labels)

            for b, l, s in zip(selected_boxes, selected_labels, scores):
                class_id = int(l)
                class_name = dhaka_ai_classes[class_id]
            
                xmin, ymin, xmax, ymax = list(map(int, b))

                if xmax > w:
                    xmax = w
                if xmin < 0:
                    xmin = 0
                if ymax > h:
                    ymax = h
                if ymin < 0:
                    ymin = 0
                score = '{:.6f}'.format(s)
                check_badbox(image_path, h, w,
                             xmin, ymin, xmax, ymax)

                result_file_writer.writerow([os.path.basename(image_path), class_name, score, xmin, ymin, xmax, ymax, image_size, image_size])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--source-dir', type=str, help='source directory of test image files')
    parser.add_argument('--class-names', type=str, help='path to text file containing names of classes in each line')
    parser.add_argument('--phi', type=int, default=4, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()
    print(opt)
    main(opt)