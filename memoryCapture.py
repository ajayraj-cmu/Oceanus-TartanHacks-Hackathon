import cv2
import time
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np
from sender import ScreenCapture  # Importing ScreenCapture from sender.py

# Configuration
TRACKED_ITEMS = ["person", "bottle", "laptop"]  # Add more items to track here
JSON_UPDATE_INTERVAL = 3  # Update JSON every 3 seconds
MODEL = "gpt-4-vision-preview"

def encode_frame_to_base64(frame):
    """Convert CV2 frame to base64 string."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_items(frame, frame_time):
    """Detect items using GPT-4V API (simulated in this example)."""
    image_base64 = encode_frame_to_base64(frame)
    
    # Simulated response for testing purposes
    response_content = '''
    {
        "person": {
            "position": [150, 200],
            "confidence": 0.95,
            "description": "Near the center of the frame"
        },
        "bottle": {
            "position": [300, 400],
            "confidence": 0.90,
            "description": "On the right side"
        }
    }
    '''
    
    try:
        detected_items = json.loads(response_content)
        formatted_items = {}
        for item, details in detected_items.items():
            if item.lower() in [t.lower() for t in TRACKED_ITEMS]:
                formatted_items[item] = {
                    "position": tuple(details["position"]),
                    "confidence": details["confidence"],
                    "timestamp": frame_time,
                    "description": details.get("description", "")
                }
        return formatted_items
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {response_content}")
        return {}

def update_json(tracking_data, filename="tracked_items.json"):
    with open(filename, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def extract_frames_from_screen(interval=3):
    screen_capture = ScreenCapture(region={"top": 100, "left": 100, "width": 640, "height": 480})
    frame_count = 0
    extracted_frames = []
    tracking_data = {}
    last_update = time.time()
    
    try:
        while True:
            frame = screen_capture.capture_frame()
            if frame is None:
                print("Failed to capture frame.")
                break

            frame_count += 1
            if frame_count % (interval * 10) == 0:  # Capture frame every `interval` seconds
                current_time = time.time()
                print(f"Processing frame at {current_time:.2f} seconds...")

                detected_items = detect_items(frame, current_time)
                frame_data = {
                    "frame_time": current_time,
                    "items": detected_items,
                    "original_frame": frame.copy()
                }

                extracted_frames.append(frame_data)
                tracking_data[str(current_time)] = detected_items

                if time.time() - last_update > JSON_UPDATE_INTERVAL:
                    update_json(tracking_data)
                    last_update = time.time()

                for item, details in detected_items.items():
                    x, y = details["position"]
                    cv2.putText(frame, f"{item} ({x},{y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, details.get("description", ""), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.imshow("Screen Capture with Annotations", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        screen_capture.release()
        update_json(tracking_data)  # Final update
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    extract_frames_from_screen(interval=3)
