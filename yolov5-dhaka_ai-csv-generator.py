import argparse
import os
import glob
import csv
import time
import cv2


def get_class_names(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def get_img_size(file):
    img = cv2.imread(file)
    assert img is not None, "Image cat not be read, path: "+file
    height, width, _ = img.shape
    return height, width


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


def process(classes):
    image_id = []
    classname = []
    score = []
    xmin = []
    xmax = []
    ymax = []
    ymin = []
    height = []
    width = []
    for file in glob.glob(os.path.join(opt.label_dir, "*.txt")):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                basename = os.path.splitext(os.path.basename(file))[0]
                image_id.append(basename + ".jpg")
                classname.append(classes[int(row[0])])
                score.append(row[1])

                x_center = float(row[2])
                y_center = float(row[3])
                box_width = float(row[4])
                box_height = float(row[5])

                h, w = get_img_size(os.path.join(
                    opt.image_dir, basename + ".jpg"))
                x_center = x_center * w
                y_center = y_center * h
                box_width = box_width * w
                box_height = box_height * h

                x_min = x_center - box_width/2
                x_max = x_center + box_width/2
                y_min = y_center - box_height/2
                y_max = y_center + box_height/2

                if x_max > w:
                    x_max = w
                if x_min < 0:
                    x_min = 0
                if y_max > h:
                    y_max = h
                if y_min < 0:
                    y_min = 0

                xmin.append(x_min)
                xmax.append(x_max)
                ymin.append(y_min)
                ymax.append(y_max)

                height.append(h)
                width.append(w)
                check_badbox(basename + ".jpg", h, w,
                             x_min, y_min, x_max, y_max)

    with open('submission_files/arafat_yolo-result_conf-{}_IOUthr-{}_{}_ac-0.0_epc-0.csv'.format(opt.conf_thres, opt.iou_thres, time.strftime("%Y-%m-%d_%H-%M-%S")), mode='w') as result_file:
        fieldnames = ['image_id', 'class', 'score', 'xmin',
                      'ymin', 'xmax', 'ymax', 'width', 'height']
        result_file_writer = csv.writer(
            result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_file_writer.writerow(fieldnames)
        for index in range(len(image_id)):
            result_file_writer.writerow([image_id[index], classname[index],
                                         score[index], xmin[index], ymin[index],
                                         xmax[index], ymax[index], height[index],
                                         width[index]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str,
                        help='source directory to read test images')
    parser.add_argument('--label-dir', type=str,
                        help='source directory to read darknet labels')
    parser.add_argument('--classname-file', type=str,
                        help='class name text file path')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()
    print(opt)
    classes = get_class_names(opt.classname_file)
    print(classes)
    process(classes)
