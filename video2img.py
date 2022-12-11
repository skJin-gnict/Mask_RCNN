import cv2
import os
import random

test_size = 0.1

os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/val', exist_ok=True)

cap = cv2.VideoCapture('assets/video.mov')

i = 0

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    if random.random() > test_size:
        cv2.imwrite(f'dataset/train/{str(i).zfill(5)}.png', img)
    else:
        cv2.imwrite(f'dataset/val/{str(i).zfill(5)}.png', img)

    i += 1
