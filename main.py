

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from PIL import Image
import torch

# Import our helper functions and path planning
from utils import (
    speak_mac,
    get_target_object,
    run_yolo_detection,
    run_depth_heatmap,
    heavy_inference,
    get_object_mask  # (dummy segmentation function)
)
from pathFind import dummy_pathfinder

# Import YOLO and Hugging Face pipeline modules
from ultralytics import YOLO
from transformers import pipeline
from segment_anything import SamPredictor, sam_model_registry

# ---------------------------
# Global Configuration & Model Initialization
# ---------------------------
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Device selection: use GPU if available, else fallback to CPU (or MPS on macOS)
device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
    else "cpu"
)

# Initialize YOLOv8 (full-resolution)
yolo_model = YOLO('yolov8n.pt')

# Initialize Depth Anything (for depth estimation)
depth_pipe = pipeline("depth-estimation",
                      model="depth-anything/Depth-Anything-V2-Small-hf",
                      device=device)

# Initialize SAM (Segment Anything)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Update with the correct path to your checkpoint
model_type = "vit_h"  # Options: "vit_h", "vit_b", etc.
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# ---------------------------
# Asynchronous Heavy Inference Setup
# ---------------------------
downscale_factor = 0.5        # Process depth and SAM on a downscaled copy
heavy_update_interval = 3     # Update heavy inference every 3 frames
frame_count = 0
last_depth_array = None       # Cached full-resolution depth map
last_ground_mask = None       # Cached full-resolution SAM ground mask
inference_future = None       # Future for asynchronous heavy inference
executor = ThreadPoolExecutor(max_workers=1)

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    global frame_count, inference_future, last_depth_array, last_ground_mask

    # Use the (dummy) ASR to get the target object name
    target_obj = get_target_object(record_duration=3)
    if not target_obj:
        target_obj = "bottle"  # fallback default
    speak_mac(f"Perfect! Let's go find your '{target_obj}'")
    
    # Set thresholds and parameters
    AREA_THRESHOLD = 2.25
    DEPTH_THRESHOLD = 30
    TIME_SPEAK_INTERVAL = 0.75
    initial_depth = {}
    initial_area = {}
    
    # Open video capture (adjust the camera index as needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    interval_time = 0
    while True:
        data = []  # List to hold detection data for path planning
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        # Resize frame to a fixed resolution for processing
        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape
        combined_mask = np.zeros_like(frame)

        # --- Create Downscaled Frame for Heavy Inference ---
        small_width = int(width * downscale_factor)
        small_height = int(height * downscale_factor)
        small_frame = cv2.resize(frame, (small_width, small_height))

        # --- Asynchronous Heavy Inference ---
        if frame_count % heavy_update_interval == 0:
            if inference_future is None or inference_future.done():
                inference_future = executor.submit(
                    heavy_inference, small_frame, width, height, depth_pipe, predictor
                )
        if inference_future is not None and inference_future.done():
            try:
                last_depth_array, last_ground_mask = inference_future.result()
            except Exception as e:
                print("Error in heavy inference:", e)
            inference_future = None
        depth_array = last_depth_array
        ground_mask = last_ground_mask

        # --- Generate Depth Heat Map Overlay ---
        depth_heat = run_depth_heatmap(depth_array, frame)
        
        # --- Run YOLO Object Detection ---
        detection_data, obstacle_mask, _, frame_with_boxes, depth_overlay = run_yolo_detection(frame, depth_array, yolo_model)
        data.extend(detection_data)
        
        # --- Compute Open (Walkable) Ground ---
        if ground_mask is not None:
            open_ground_mask = cv2.bitwise_and(ground_mask, cv2.bitwise_not(obstacle_mask))
        else:
            open_ground_mask = np.zeros((height, width), dtype=np.uint8)
        green_overlay = np.zeros_like(frame)
        green_overlay[open_ground_mask == 255] = (0, 255, 0)
        overlayed_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.5, 0)

        # --- Display Windows ---
        # 1. Original frame with YOLO boxes and green overlay for open ground.
        cv2.imshow("YOLO with Walkable Ground Overlay", overlayed_frame)
        # 2. Depth heat map with YOLO boxes.
        cv2.imshow("Depth Heat Map with YOLO", depth_overlay)
        # 3. Raw SAM ground mask.
        if ground_mask is not None:
            cv2.imshow("SAM Ground Mask", ground_mask)
        else:
            cv2.imshow("SAM Ground Mask", np.zeros((height, width), dtype=np.uint8))

        # --- Dummy Path Planning ---
        move, location = dummy_pathfinder(data, target_obj)
        # (You can add logic to compare initial depth/area thresholds here.)
        if location==None : speak_mac("Object not found. Try checking your pockets or another room.")
        
        interval_time += time.time() - start_time
        if interval_time > TIME_SPEAK_INTERVAL:
            speak_mac(move)
            interval_time = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()
