from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

ROOT_DIR = '/home/ubuntu/dev/skjin/tf25_Mask_RCNN'

sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

import os
import sys
import json
import numpy as np
import datetime
#import mrcnn.utils
import mrcnn.config
#import mrcnn.model

import glob
import skimage.draw
from tqdm import tqdm


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class ParkConfig(mrcnn.config.Config): #옵션 설정.
    NAME = "bbox"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    #LEARNING_RATE = 0.001

    NUM_CLASSES = 1 + 38  # Background + 40객체

    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
parkConfig = ParkConfig()
parkConfig.display()


class ParkDataset(utils.Dataset):
#데이터셋 가져오는곳.
    def load_park(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("bbox", 1, "Car")
        self.add_class("bbox", 2, "Van")
        self.add_class("bbox", 3, "Other Vehicle")
        self.add_class("bbox", 4, "Motorbike")
        self.add_class("bbox", 5, "Bicycle")
        self.add_class("bbox", 6, "Electric Scooter")
        self.add_class("bbox", 7, "Adult")
        self.add_class("bbox", 8, "Child")
        self.add_class("bbox", 9, "Stroller")
        self.add_class("bbox", 10, "Shopping Cart")
        self.add_class("bbox", 11, "Gate Arm")
        self.add_class("bbox", 12, "Parking Block")
        self.add_class("bbox", 13, "Speed Bump")
        self.add_class("bbox", 14, "Traffic Pole")
        self.add_class("bbox", 15, "Traffic Cone")
        self.add_class("bbox", 16, "Traffic Drum")
        self.add_class("bbox", 17, "Traffic Barricade")
        self.add_class("bbox", 18, "Cylindrical Bollard")
        self.add_class("bbox", 19, "U-shaped Bollard")
        self.add_class("bbox", 20, "Other Road Barriers")
        self.add_class("bbox", 21, "No Parking Stand")
        self.add_class("bbox", 22, "Adjustable Parking Pole")
        self.add_class("bbox", 23, "Waste Tire")
        self.add_class("bbox", 24, "Planter Barrier")
        self.add_class("bbox", 25, "Water Container")
        self.add_class("bbox", 26, "Movable Obstacle")
        self.add_class("bbox", 27, "Barrier Gate")
        self.add_class("bbox", 28, "Electric Car Charger")
        self.add_class("bbox", 29, "Parking Meter")
        self.add_class("bbox", 30, "Parking Sign")
        self.add_class("bbox", 31, "Traffic Light")
        self.add_class("bbox", 32, "Pedestrian Light")
        self.add_class("bbox", 33, "Street Sign")
        self.add_class("bbox", 34, "Disabled Parking Space")
        self.add_class("bbox", 35, "Pregnant Parking Space")
        self.add_class("bbox", 36, "Electric Car Parking Space")
        self.add_class("bbox", 37, "Two-wheeled Vehicle Parking Space")
        self.add_class("bbox", 38, "Other Parking Space")
        

        # Train or validation dataset?
        assert subset in ["Training", "Validation", "Test"]
        label_dir = os.path.join(dataset_dir,"2_라벨링데이터", subset)
        data_dir = os.path.join(dataset_dir,"1_원천데이터", subset)
        bbox_file_list = []
        print(datetime.datetime.now(), "check start")
    
        # Load annotations
        json_list = glob.glob(os.path.join(label_dir, "*/*/*/label/*.json"))
        annotations_b = []
        annotations_p = []
        
        print(datetime.datetime.now(), "list check ok")
        for json_file in tqdm(json_list):
            with open(json_file, 'rb') as f:
                data = json.load(f)
                if len(data['bbox2d'])!=0:
                    annotations_b.append([json_file,data])
        print(datetime.datetime.now(), "list load ok")
        # Add images
        for a in tqdm(annotations_b):

            bboxs = [b['bbox'] for b in a[1]['bbox2d']]
            name = [b['name'] for b in a[1]['bbox2d']]
            name_dict ={
                "Car" : 1,
                "Van" : 2,
                "Other Vehicle" : 3, 
                "Motorbike" : 4,
                "Bicycle" : 5,
                "Electric Scooter" : 6,
                "Adult" : 7,
                "Child" : 8,
                "Stroller" : 9,
                "Shopping Cart" : 10,
                "Gate Arm" : 11,
                "Parking Block" : 12,
                "Speed Bump" : 13,
                "Traffic Pole" : 14,
                "Traffic Cone" : 15,
                "Traffic Drum" : 16,
                "Traffic Barricade" : 17,
                "Cylindrical Bollard" : 18,
                "U-shaped Bollard" : 19,
                "Other Road Barriers" : 20,
                "No Parking Stand" : 21, 
                "Adjustable Parking Pole" : 22,
                "Waste Tire" : 23,
                "Planter Barrier" : 24,
                "Water Container" : 25,
                "Movable Obstacle" : 26,
                "Barrier Gate" : 27,
                "Electric Car Charger" : 28,
                "Parking Meter" : 29,
                "Parking Sign" : 30,
                "Traffic Light" : 31,
                "Pedestrian Light" : 32,
                "Street Sign" : 33,
                "Disabled Parking Space" : 34,
                "Pregnant Parking Space" : 35,
                "Electric Car Parking Space" : 36,
                "Two-wheeled Vehicle Parking Space" : 37,
                "Other Parking Space" : 38
                }

            num_ids = []

            for i,n in enumerate(name) : 
                if n in name_dict :
                    num_ids.append(name_dict[n])
                    
                else : 
#                     print(f'{a[0]} 파일 {n} 객체 오류')
                    del bboxs[i]

            a_img = a[0].split('/')
            file_name = f'{a_img[-1][:-5]}.jpg'
            image_path = os.path.join(data_dir,a_img[-5],a_img[-4],a_img[-3],"Camera",file_name)
            
            if os.path.exists(image_path):
                try:
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "bbox",
                        image_id=f"{a_img[-4]}/{a_img[-3]}_b/{file_name}",  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        bboxs=bboxs,
                        num_ids=num_ids)
                    bbox_file_list.append(image_path)
                    #print(f'{image_path} 불러오기 성공하였슴')
                except :
                    #print(f'{image_path} 불러오기 실패')
                    pass
#         import csv
        
#         with open(f'bboxlistfile{subset}.csv','w') as f :
#             writer = csv.writer(f)
#             writer.writerow(bbox_file_list)
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "bbox":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        if info["source"] != "bbox":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["bboxs"])],
                        dtype=np.uint8)
        if 'bboxs' in info :
            bboxs =info["bboxs"]

            for i in range(len(bboxs)) :
                bbox = bboxs[i]
                row_s, row_e = int(bbox[1]), int(bbox[3])
                col_s, col_e = int(bbox[0]), int(bbox[2])
                mask[row_s:row_e, col_s:col_e, i] = 1

        if 'polygons' in info:
            
            polygons = info['polygons']
            for i in range(len(polygons)) :
                polygon = polygons[i]
                all_points_y = []
                all_points_x = []
                for po in polygon :
                    x = po[0]
                    y = po[1]
                    all_points_x.append(float(x))
                    all_points_y.append(float(y))
                rr, cc = skimage.draw.polygon(all_points_y,all_points_x)

                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool_), num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bbox":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


#=============main========
dataset_train = ParkDataset()
dataset_train.load_park(dataset_dir='/data/Dataset2d_Final', subset = "Training")
dataset_train.prepare()
print('Train', len(dataset_train.image_ids))


dataset_val = ParkDataset()
dataset_val.load_park(dataset_dir='/data/Dataset2d_Final', subset = "Validation")
dataset_val.prepare()
print('Validation', len(dataset_val.image_ids))

image_ids = np.random.choice(dataset_train.image_ids, 4)


for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

model = modellib.MaskRCNN(
    mode="training",
    config=parkConfig,
    model_dir=MODEL_DIR)

model.load_weights(filepath='/data/Mask_RCNN/mask_rcnn_bbox_0051.h5', 
                   by_name=True)

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

model.train(
    dataset_train,
    dataset_val, 
    learning_rate=parkConfig.LEARNING_RATE,
    epochs=50, 
    layers="all")


model_path = 'trained_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)