import mss
import numpy as np
import cv2

class ScreenCapture:
    def __init__(self, region=None):
        """
        Initialize the ScreenCapture class.
        :param region: Tuple (left, top, width, height) specifying the screen region to capture.
        """
        self.sct = mss.mss()
        self.region = region if region else self._get_default_region()
    
    def _get_default_region(self):
        """
        Set a default region for screen capture if none is provided.
        This region is a placeholder representing a typical WhatsApp call window.
        Modify the coordinates as needed to match your WhatsApp window.
        """
        return {"top": 100, "left": 100, "width": 640, "height": 480}
    
    def capture_frame(self):
        """
        Capture a single frame from the specified screen region.
        :return: Frame as a numpy array in BGR format.
        """
        frame = self.sct.grab(self.region)
        img = np.array(frame)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def release(self):
        """Release resources (not strictly necessary with mss)."""
        self.sct.close()


# For testing this script independently
if __name__ == "__main__":
    # Create the screen capture object with a specific region
    screen_capture = ScreenCapture(region={"top": 100, "left": 100, "width": 640, "height": 480})
    
    print("Capturing frames from the specified screen region...")
    
    while True:
        frame = screen_capture.capture_frame()
        
        # Display the frame in a window
        cv2.imshow("Screen Capture (WhatsApp Call)", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    screen_capture.release()
    cv2.destroyAllWindows()
