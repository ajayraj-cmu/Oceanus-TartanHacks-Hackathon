import cv2
import time
import json
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np
from sender import ScreenCapture  # Importing ScreenCapture from sender.py
import openai  # Ensure you have the OpenAI client installed

# Configuration
TRACKED_ITEMS = ["person", "bottle", "laptop"]  # Add more items to track here
JSON_UPDATE_INTERVAL = 3  # Update JSON every 3 seconds
MODEL = "gpt-4-vision-preview"
openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key

def encode_frame_to_base64(frame):
    """Convert CV2 frame to base64 string."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_items_with_gpt4(frame, frame_time):
    """Send frame to GPT-4 and get object descriptions."""
    image_base64 = encode_frame_to_base64(frame)
    prompt = (
        f"If any of the following objects are located in this image: {', '.join(TRACKED_ITEMS)}, "
        "please describe their position in relation to other objects in the setting. "
        "Be descriptive enough such that a blind individual could use your response as a guide to locate the object. "
        "If none of the objects are in the image, just respond with 'None'."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a computer vision assistant."},
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": f"Here is the image: data:image/jpeg;base64,{image_base64}"
                }
            ],
            max_tokens=1000
        )

        response_text = response["choices"][0]["message"]["content"]
        print(f"GPT-4 Response: {response_text}")

        if response_text.strip().lower() == "none":
            return {}

        # Parse response into JSON if possible
        return {"description": response_text, "timestamp": frame_time}

    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return {}

def update_json(tracking_data, filename="tracked_items.json"):
    """Write the tracking data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def extract_frames_from_screen(interval=3):
    """Capture frames from the screen and process them using GPT-4."""
    screen_capture = ScreenCapture(region={"top": 100, "left": 100, "width": 640, "height": 480})
    frame_count = 0
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

                detected_items = detect_items_with_gpt4(frame, current_time)
                tracking_data[str(current_time)] = detected_items

                # Update JSON periodically
                if time.time() - last_update > JSON_UPDATE_INTERVAL:
                    update_json(tracking_data)
                    last_update = time.time()

                # Display the frame with a label indicating it has been processed
                cv2.putText(frame, "Frame Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
