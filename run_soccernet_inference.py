import cv2
import os
from ultralytics import YOLO

def run_soccernet_inference(image_path, weights_path):
    # Load the YOLO model
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    model = YOLO(weights_path)
    
    # Run inference on the image
    # Use device='cpu' to avoid CUDA errors
    results = model(image_path, conf=0.25, device='cpu')[0]

    # Plot the results on the image
    annotated_img = results.plot()

    # Save the annotated image
    output_path = "keypoint_detection_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    
    print(f"Inference complete. Results saved to {output_path}")

if __name__ == "__main__":
    # Path to the yolo26 soccernet weights
    weights = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\weights\keypoint_detection_online_model.pt"
    
    # Path to a test image
    image = r"c:\Users\jessi\Downloads\tactical_view_image.webp"
    
    run_soccernet_inference(image, weights)
