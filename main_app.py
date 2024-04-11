import streamlit as st
import os
import uuid
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from collections import defaultdict

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)
model.eval()

def calculate_iou(box1, box2):
    # Calculate the intersection area
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Calculate the union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou

def non_max_suppression(selected_data, iou_threshold):
    sorted_boxes = sorted(selected_data, key=lambda box: box[4], reverse=True)
    selected_boxes = []

    for box in sorted_boxes:
        if not any(
                calculate_iou(box, selected_box) > iou_threshold
                for selected_box in selected_boxes
        ):
            selected_boxes.append(box)

    return selected_boxes

def detect_objects(img):
    results = model([img])
    data = results.pandas().xyxy[0]
    selected_data = data[["xmin", "ymin", "xmax", "ymax", "name"]].values.tolist()

    iou_threshold = 0.8
    selected_bounding_boxes = non_max_suppression(selected_data, iou_threshold)

    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    font_size = 12
    font = ImageFont.truetype("ProximaNova-Regular.otf", font_size)

    for index, annotation in enumerate(selected_bounding_boxes, start=1):
        x_min, y_min, x_max, y_max, class_name = annotation
        draw.rectangle([x_min, y_min, x_max, y_max], outline="yellow", width=1)
        draw.text((x_min, y_min), f"{index}: {class_name}", fill="white", font=font)

    return img_with_boxes, selected_bounding_boxes

def main():
    st.title("Object Detection and Class Update")

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if file is not None:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        st.image(img, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Objects'):
            img_with_boxes, selected_bounding_boxes = detect_objects(img)
            st.image(img_with_boxes, caption='Detected Objects', use_column_width=True)

            class_counts = defaultdict(int)
            total_annotations = len(selected_bounding_boxes)

            for entry in selected_bounding_boxes:
                class_label = entry[-1]
                class_counts[class_label] += 1

            class_percentages = {
                class_name: round(count / total_annotations * 100, 2)
                for class_name, count in class_counts.items()
            }

            st.write("Class Counts:", class_counts)
            st.write("Class Percentages:", class_percentages)

            # Class update
            st.subheader("Update Class")
            index_dt = []
            for index, annotation in enumerate(selected_bounding_boxes, start=1):
                entry = {"index": index, "class": annotation[4], "BBox": annotation[:4]}
                index_dt.append(entry)

            new_index_dt = st.text_area("Annotations", value=json.dumps(index_dt), height=200)
            image_with_updated_class = Image.open(io.BytesIO(img_bytes))
            if st.button('Update Class'):
                try:
                    new_index_dt = json.loads(new_index_dt)
                    draw = ImageDraw.Draw(image_with_updated_class)
                    font_size = 15
                    font = ImageFont.truetype("ProximaNova-Regular.otf", font_size)

                    for item in new_index_dt:
                        index = item["index"]
                        class_name = item["class"]
                        bbox_values = item["BBox"]
                        x_min, y_min, x_max, y_max = bbox_values

                        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)
                        text = f"{index}: {class_name}"
                        text_bbox = draw.textbbox((x_min, y_min - font_size), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)
                        draw.rectangle(
                            [x_min, y_min - text_height, x_min + text_width, y_min],
                            fill="green",
                        )
                        draw.text((x_min, y_min - text_height), text, font=font, fill="white")

                    st.image(image_with_updated_class, caption='Updated Class', use_column_width=True)

                except (json.JSONDecodeError, ValueError) as e:
                    st.error(f"Invalid index_dt format {str(e)}")

if __name__ == "__main__":
    main()