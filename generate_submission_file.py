import cv2
import json
import numpy as np
import os
import time
import glob
import csv
import sys

import  tensorflow as tf

from model import efficientdet
from utils import preprocess_image, postprocess_boxes


def main(args=None):
    if len(sys.argv) is 3:
        model_path = str(sys.argv[1])
        image_data = os.path.join(str(sys.argv[2]), "*.jpg")
        print(model_path)
        print(image_data)
    else:        
        print("Pass model path and image data path in respectively as command line argument")
        exit()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 4
    weighted_bifpn = False
    model_path = model_path
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    coco_classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    coco_num_classes = 90

    # Dhaka-ai classes
    dhaka_ai_classes = {
        0:  'ambulance',
        1:  'auto rickshaw',
        2:  'bicycle',
        3:  'bus',
        4:  'car',
        5:  'garbagevan',
        6:  'human hauler',
        7:  'minibus',
        8:  'minivan',
        9:  'motorbike',
        10: 'pickup',
        11: 'army vehicle',
        12: 'policecar',
        13: 'rickshaw',
        14: 'scooter',
        15: 'suv',
        16: 'taxi',
        17: 'three wheelers (CNG)',
        18: 'truck',
        19: 'van',
        20: 'wheelbarrow'
    }
    dhaka_ai_num_classes = 21

    score_threshold = 0.15
    # colors = [np.random.randint(0, 256, 3).tolist() for _ in range(dhaka_ai_num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=dhaka_ai_num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)

<<<<<<< HEAD
    with open('submission_files/result-{}.csv'.format(time.strftime("%Y-%m-%d_%H-%M-%S")), mode='w') as result_file:
=======
    with open('submission_files/result-{}.csv'.format(time.strftime("%Y-%m-%d_%H-%M-%S")), mode='w') as result_file:
>>>>>>> tensorflow-latest
        fieldnames = ['image_id', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']
        result_file_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_file_writer.writerow(fieldnames)
        for image_path in glob.glob(image_data):
            image = cv2.imread(image_path)
            src_image = image.copy()
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

            # select indices which have a score above the threshold
            indices = np.where(scores[:] > score_threshold)[0]

            # select indices which have a score above the threshold
            # indices = np.where(scores[:] > score_threshold)[0]

            # select those detections
            # boxes = boxes[indices]
            # labels = labels[indices]

            
            selected_indices = tf.image.non_max_suppression(
                boxes, scores, 50, iou_threshold=0.4, score_threshold=0.30)
            selected_boxes = tf.gather(boxes, selected_indices)
            selected_labels = tf.gather(labels, selected_indices)
            selected_boxes = tf.Session().run(selected_boxes)
            selected_labels = tf.Session().run(selected_labels)
            # boxes = boxes[selected_indices]
            # labels = labels[selected_indices]

            # store_csv(result_file, src_image, boxes, scores, labels, colors, dhaka_ai_classes)
            for b, l, s in zip(selected_boxes, selected_labels, scores):
                class_id = int(l)
                class_name = dhaka_ai_classes[class_id]
            
                xmin, ymin, xmax, ymax = list(map(int, b))
                score = '{:.2f}'.format(s)
                # color = colors[class_id]
                # label = '-'.join([class_name, score])
            

                # ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
                # cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
                # cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                result_file_writer.writerow([os.path.basename(image_path), class_name, score, xmin, ymin, xmax, ymax, image_size, image_size])



if __name__ == '__main__':
    main()
