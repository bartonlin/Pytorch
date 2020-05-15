from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
import numpy

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

colors = [(0, 127, 255), (0, 255, 255), (0, 255, 127), (0, 255, 0), (127, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0), (255, 0, 127), (255, 0, 255)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--video_file", type=str, default="", help="path to video dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_file_name = opt.video_file.split(".")[0].split("/")[-1]
    os.makedirs("output/" + result_file_name , exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    
    if opt.video_file:
        vidcap = cv2.VideoCapture(opt.video_file)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #data/video/test.mp4
        save_frame_path = "./data/" + opt.video_file.split(".")[0].split("/")[-1] + "_frames"
        #print(save_frame_path)
        os.makedirs(save_frame_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(save_frame_path + "/%d.png" % count, image)
            success,image = vidcap.read()
            count += 1
        img_frame_path = save_frame_path
        img_frame_size = 416
    else:
        img_frame_path = opt.image_folder
        img_frame_size = opt.img_size
            
    dataloader = DataLoader(
        ImageFolder(img_frame_path, img_size=img_frame_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    videoWriter = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    # Bounding-box colors

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        path = path.replace("\\", "/")  # for windows
        # Create plot
        image = cv2.imread(path)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, image.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                color = colors[int(cls_pred)%6]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
                cv2.putText(image, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3, color, 1, cv2.LINE_AA)
        
        filename = path.split("/")[-1].split(".")[0]

        cv2.imshow('frame', image)
        cv2.waitKey(1000//int(fps) + 1)
        videoWriter.write(image)
        cv2.imwrite(f"output\{result_file_name}\{filename}.png", image)
        
    cv2.destroyAllWindows()
    videoWriter.release

