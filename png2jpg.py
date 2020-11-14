#!/usr/bin/env python

from glob import glob
import os
import cv2
import sys

dir=sys.argv[1]

print("Converting png to jpg in dir:", dir)


pngs = glob(os.path.join(dir, "*.png"))
print("Total pngs:", len(pngs))

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)



pngs = glob(os.path.join(dir, "*.PNG"))

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)

