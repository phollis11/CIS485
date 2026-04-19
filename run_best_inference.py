import cv2
import os
from ultralytics.nn.autobackend import AutoBackend

def run_inference():
    model_path = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\weights\yolo26n_field_detection_model.pt"
    video_path = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\raw\prem_vid_short.mp4"
    output_dir = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\inference_output"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print("📦 Loading model (safe backend)...")
    model = AutoBackend(model_path, device="cpu")  # change to "cuda" if GPU

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(output_dir, "output.mp4")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0

    print("🎥 Running inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Run inference (RAW)
        preds = model(frame)

        # NOTE: preds format is raw — we won't rely on Ultralytics postprocess
        # Just display frame count for now
        cv2.putText(frame, f"Frame: {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame to video
        out.write(frame)

        if frame_id % 30 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    out.release()

    print("\n✅ Inference complete!")
    print(f"📁 Output saved to:\n{output_path}")


if __name__ == "__main__":
    run_inference()