import cv2
import torch
import os
import numpy as np
from .manager import GMCLink

def run_visualization(frame_folder, text_prompt, text_embedding):
    """
    Visualizes GMC-Link's reasoning over a sequence of frames.
    """
    # 1. Initialize the Master Manager
    linker = GMCLink()
    
    # Load your trained weights if you have them
    # linker.aligner.load_state_dict(torch.load("gmc_link_weights.pth"))
    linker.aligner.eval()

    # Get sorted frame list
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(('.jpg', '.png'))])
    
    # Mock Track: Let's pretend we have one car moving across the screen
    # In a real scenario, these centroids come from your detector (YOLO/ByteTrack)
    track_id = 1
    current_x, current_y = 100, 300 

    print(f"Processing {len(frame_files)} frames for prompt: '{text_prompt}'")

    for i, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None: continue

        # --- SIMULATE TRACK DATA (For Testing) ---
        # We simulate the car moving right (dx=5 pixels per frame)
        prev_x, prev_y = current_x, current_y
        current_x += 5 
        
        # Create a mock track object compatible with our manager.py logic
        class MockTrack:
            def __init__(self, tid, curr, prev):
                self.id = tid
                self.centroid = np.array(curr)
                self.prev_centroid = np.array(prev)
        
        active_tracks = [MockTrack(track_id, [current_x, current_y], [prev_x, prev_y])]

        # --- GMC-LINK PROCESSING ---
        # This calls core.py (GMC), utils.py (Compensation), and alignment.py (Reasoning)
        scores_dict = linker.process_frame(frame, active_tracks, text_embedding)
        score = scores_dict.get(track_id, 0.0)

        # --- VISUALIZATION OVERLAY ---
        # Draw the object
        cv2.circle(frame, (int(current_x), int(current_y)), 10, (0, 255, 0), -1)
        
        # Draw the score and prompt info
        color = (0, 255, 0) if score > 0.5 else (0, 0, 255) # Green if matches, Red if not
        cv2.putText(frame, f"Prompt: {text_prompt}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"ID {track_id} Alignment: {score:.2f}", (int(current_x) - 40, int(current_y) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame
        cv2.imshow("GMC-Link Visualization", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # 1. Provide your frames folder
    folder = "path/to/your/frames" 
    
    # 2. Provide a mock embedding (768-dim)
    # In reality, you'd use: text_embedding = clip_model.encode("moving right")
    mock_text_embedding = torch.randn(1, 768) 
    
    run_visualization(folder, "moving right", mock_text_embedding)