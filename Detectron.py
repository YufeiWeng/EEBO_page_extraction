# %% [markdown]
# # Detectron2 Beginner's Tutorial
# 
# <img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">
# 
# Welcome to detectron2! This is the official colab tutorial of detectron2. Here, we will go through some basics usage of detectron2, including the following:
# * Run inference on images or videos, with an existing detectron2 model
# * Train a detectron2 model on a new dataset
# 
# You can make a copy of this tutorial by "File -> Open in playground mode" and make changes there. __DO NOT__ request access to this tutorial.
# 

# %% [markdown]
# # Install detectron2

# %%
# !python -m pip install pyyaml==5.1
# import sys, os, distutils.core
# # Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# # See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
# sys.path.insert(0, os.path.abspath('./detectron2'))

# # Properly install detectron2. (Please do not install twice in both ways)
# # !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# %%
import sys
print(sys.executable)

# %%
import numpy as np
m = np.zeros(3)
m

# %%
import torch, detectron2
# !nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# %%
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from tqdm import tqdm
import os
import json
import cv2
import random
# from google.colab.patches import cv2_imshow
from IPython.display import display, Image
import io

def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    display(Image(data=buffer))

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# %% [markdown]
# # Train on a custom dataset

# %% [markdown]
# In this section, we show how to train an existing detectron2 model on a custom dataset in a new format.
# 
# We use [the balloon segmentation dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
# which only has one class: balloon.
# We'll train a balloon segmentation model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.
# 
# Note that COCO dataset does not have the "balloon" category. We'll be able to recognize this new class in a few minutes.
# 
# ## Prepare the dataset

# %% [markdown]
# Register the balloon dataset to detectron2, following the [detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).
# Here, the dataset is in its custom format, therefore we write a function to parse it and prepare it into detectron2's standard format. User should write such a function when using a dataset in custom format. See the tutorial for more details.
# 

# %%
height, width = cv2.imread("/trunk/shared/pair_data/A36588/00013.000.001.tif").shape[:2]
print(height, width)

# %%
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
from pathlib import Path
import os
import cv2
import json
import glob
from detectron2.structures import BoxMode

def find_image_file(base_path, file_name_without_extension):
    # Define the possible image extensions
    image_extensions = ['.jpg', '.png', '.JPG', '.tif']
    
    # Search for image files with the specified base name and extensions
    for extension in image_extensions:
        # Build the path for the image file
        image_path = os.path.join(base_path, file_name_without_extension + extension)
        if os.path.exists(image_path):
            return image_path  # Return the image path if file exists

    return None  # If no image file is found

def process_annotations(annotations):
    objs = []
    for anno in annotations:
        px = [point[0] for point in anno['polygon']]
        py = [point[1] for point in anno['polygon']]
        poly = [(x, y) for x, y in zip(px, py)]
        poly_flatten = [p for x in poly for p in x]

        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly_flatten],
            "category_id": 0,  # Assuming you have only one category
            "confidence": anno['confidence']
        }
        objs.append(obj)
    return objs

def get_custom_dicts(json_dir, image_dir):
    print("json_dir: ", json_dir)
    print("image_dir: ", image_dir)
    dataset_dicts = []
    json_files = list(Path(json_dir).rglob('*.json'))  # Get the list of JSON files

    # Use tqdm to show a progress bar
    for json_file_path in tqdm(json_files, desc='Processing JSON files'):
        with open(str(json_file_path), 'r') as f:
            annotations_data = json.load(f)
        
        record = {}
        file_name_without_extension = json_file_path.stem

        # Build the path to the image by using the base image directory and the image filename
        image_file = find_image_file(image_dir, file_name_without_extension)
        if image_file is None:
            print(f"No image found for {json_file_path.name}, skipping.")
            continue
        
        # Read image to get its width and height
        image = cv2.imread(image_file)
        height, width = image.shape[:2]

        # Create the record dict
        record["file_name"] = image_file
        record["image_id"] = file_name_without_extension
        record["height"] = height
        record["width"] = width

        # Process the annotations
        record["annotations"] = process_annotations(annotations_data)
        dataset_dicts.append(record)
    
    return dataset_dicts

# Usage
# img_dir = "/trunk2/yufei/datasets/jsons"
# base_path = '/trunk2/yufei/datasets/pictures/'
# dataset_dicts = get_custom_dicts(img_dir, base_path)

# Assuming your directories are named "train", "val", and "test" under "/trunk2/yufei/datasets/pictures/"
# for d in ["train", "val", "test"]:
for d in ["train", "validation"]:
    DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_custom_dicts(f"/trunk2/yufei/summer24/EEBO_page_extraction/dataset/jsons/{d}", f"/trunk2/yufei/summer24/EEBO_page_extraction/dataset/pictures/{d}"))
    MetadataCatalog.get("my_dataset_" + d).set(thing_classes=["my_class"])

# Now you have metadata for each of the splits
train_metadata = MetadataCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_validation")
# test_metadata = MetadataCatalog.get("my_dataset_test")



# %% [markdown]
# To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
# 
# 

# %%
# Usage
img_dir = "/trunk2/yufei/summer24/EEBO_page_extraction/dataset/jsons/train"
base_path = '/trunk2/yufei/summer24/EEBO_page_extraction/dataset/pictures/train'
dataset_dicts = get_custom_dicts(img_dir, base_path)

# dataset_dicts = get_custom_dicts(f"/trunk2/yufei/datasets/jsons/train", f"/trunk2/yufei/datasets/pictures/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])

# %% [markdown]
# ## Train!
# 
# Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~2 minutes to train 300 iterations on a P100 GPU.
# 

# %%
from detectron2.engine import DefaultTrainer
print(torch.cuda.device_count(), "GPUs")
trainFlag = True
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

if trainFlag:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# %%
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

# %% [markdown]
# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:
# 
# 

# %%
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# %% [markdown]
# Then, we randomly select several samples to visualize the prediction results.

# %%
import json
import numpy as np
import cv2
import torch
from pathlib import Path

# Function to save the prediction outputs to JSON files
def save_text_line_to_json(outputs, image_path, output_dir):
    """Converts prediction output to a JSON format and saves it to a file."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Initialize the record dictionary to store image text_line
    record = {
        "img_size": outputs['instances'].image_size,
        "text_line": []
    }

    # Extract relevant data from the outputs
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()
    classes = outputs['instances'].pred_classes.cpu().numpy()

    if outputs['instances'].has('pred_masks'):
        # Convert binary masks to boolean arrays
        masks = outputs['instances'].pred_masks.cpu().numpy()

    # Iterate over each instance prediction
    for idx in range(len(scores)):
        # Create a list to store all points from all contours of the current mask
        all_points = []

        if outputs['instances'].has('pred_masks'):
            mask = masks[idx].astype('uint8')
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate over all contours and extend the all_points list with points from each contour
            for contour in contours:
                # Flatten the contour array and extend all_points
                all_points.extend(contour.squeeze(axis=1).tolist())

        record['text_line'].append({
            "confidence": float(scores[idx]),
            "polygon": all_points  # Use the flattened list of points
        })

    # Construct file path to save the JSON data
    base_name = Path(image_path).stem
    json_file_name = f"{base_name}.json"
    json_path = os.path.join(output_dir, json_file_name)

    # Write data to JSON file
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=4)
    
    print(f"text_line saved to {json_path}")
    return json_path

# # %%
# output_dir = "/trunk2/yufei/arabic_ms_data_to_teklia/teklia/prediction"
# img_dir = "/trunk2/yufei/datasets/jsons/test"
# base_path = '/trunk2/yufei/datasets/pictures/test'
# dataset_dicts = get_custom_dicts(img_dir, base_path)

# # Assuming dataset_dicts is already defined and filled
# json_img_record = {}
# for d in dataset_dicts:
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
    
#     # Save the outputs to JSON
#     save_name_json = save_text_line_to_json(outputs, d["file_name"], output_dir)
#     json_img_record[save_name_json] = d["file_name"]
# json.dump(json_img_record, open("json_img_path_mapping_2.json", "w"))

# # %%
# from detectron2.utils.visualizer import ColorMode
# img_dir = "/trunk2/yufei/datasets/jsons/val"
# base_path = '/trunk2/yufei/datasets/pictures/val'
# dataset_dicts = get_custom_dicts(img_dir, base_path)
# # dataset_dicts = get_custom_dicts("balloon/val")
# for d in random.sample(dataset_dicts, 1):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=train_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])

# # %% [markdown]
# # We can also evaluate its performance using AP metric implemented in COCO API.
# # This gives an AP of ~70. Not bad! （Yufei's code ends here.）

# # %%
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("balloon_valll", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "balloon_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`

# # %% [markdown]
# # # Other types of builtin models
# # 
# # We showcase simple demos of other types of models below:

# # %%
# # Inference with a keypoint detection model
# cfg = get_cfg()   # get a fresh new config
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])

# # %%
# # Inference with a panoptic segmentation model
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# predictor = DefaultPredictor(cfg)
# panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
# cv2_imshow(out.get_image()[:, :, ::-1])

# # %% [markdown]
# # # Run panoptic segmentation on a video

# # %%
# # This is the video we're going to process
# from IPython.display import YouTubeVideo, display
# video = YouTubeVideo("ll8TgCZ0plk", width=500)
# display(video)

# # %%
# # Install dependencies, download the video, and crop 5 seconds for processing
# !pip install youtube-dl
# !youtube-dl https://www.youtube.com/watch?v=ll8TgCZ0plk -f 22 -o video.mp4
# !ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4

# # %%
# # Run frame-by-frame inference demo on this video (takes 3-4 minutes) with the "demo.py" tool we provided in the repo.
# !git clone https://github.com/facebookresearch/detectron2
# # Note: this is currently BROKEN due to missing codec. See https://github.com/facebookresearch/detectron2/issues/2901 for workaround.
# %run detectron2/demo/demo.py --config-file detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv \
#   --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl

# # %%
# # Download the results
# from google.colab import files
# files.download('video-output.mkv')

# # %%


# # %%


# # %%



