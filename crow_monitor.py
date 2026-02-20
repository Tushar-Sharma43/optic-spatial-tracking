import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Initialize YOLOv8 Nano model (highly optimized for real-time inference)
    model = YOLO('yolov8n.pt')
    
    # 2. Setup video capture (0 for webcam)
    cap = cv2.VideoCapture(0) 
    
    # --- DYNAMIC RESOLUTION HANDLING ---
    # Fetch the actual resolution of your active camera
    cam_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fallback in case the camera hasn't initialized its resolution properly yet
    if cam_width == 0 or cam_height == 0:
        cam_width, cam_height = 1280, 720
        
    # Create a dynamic monitoring zone: The entire bottom half of the screen
    zone = np.array([
        [0, cam_height // 2],          # Top-Left 
        [cam_width, cam_height // 2],  # Top-Right 
        [cam_width, cam_height],       # Bottom-Right 
        [0, cam_height]                # Bottom-Left 
    ], np.int32)
    # -----------------------------------
    
    # Set to store unique tracked IDs that enter our specific zone
    unique_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Run YOLOv8 tracking with ByteTrack, filtering for 'person' (class 0)
        # device="mps" explicitly leverages hardware acceleration for maximum frame rates
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[0], 
            device="mps", 
            verbose=False
        )
        
        # 4. Process tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() 
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                # EXPLICITLY cast coordinates to standard Python integers
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                
                # Calculate bounding box center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 
                
                # Check if the person's center point is inside the defined monitoring zone
                if cv2.pointPolygonTest(zone, (cx, cy), False) >= 0:
                    unique_ids.add(track_id)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
                
                # Draw the bounding box and the persistent ByteTrack ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Visual Overlays for the UI
        # Draw the dynamically sized monitoring zone
        cv2.polylines(frame, [zone], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Display the real-time analytics metric
        cv2.putText(frame, f"Unique People in Zone: {len(unique_ids)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the output
        cv2.imshow("Crowd Monitoring System", frame)
        
        # Press 'q' to cleanly exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()