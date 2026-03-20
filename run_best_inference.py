from ultralytics import YOLO
import os

def run_inference():
    # Updated paths for organized project structure
    model_path = r"C:\Users\pholl\OneDrive\Desktop\_pycache_\CIS 485\weights\yolo26_soccernet_best.pt"
    video_path = r"C:\Users\pholl\OneDrive\Desktop\_pycache_\CIS 485\samples\raw\prem_vid_short.mp4"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {video_path}")
    # Run inference and save results
    results = model.predict(source=video_path, save=True, project="inference_results", name="YOLO26n_A100GPU3, test", exist_ok=True)
    
    print("\nInference complete!")
    if results:
        print(f"Results saved to: {results[0].save_dir}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_inference()
