from ultralytics import YOLO
import os

def train_defect_detector():
    print("🚀 Initializing YOLOv8 Nano model for edge deployment...")
    
    # Load a pretrained nano model (fastest for edge devices)
    model = YOLO('yolov8n.pt') 

    print("📸 Starting training on BDFD9 Dataset...")
    
    # Train the model
    results = model.train(
        data='bdfd9.yaml',      # Path to our configuration file
        epochs=20,              # Start with 20 epochs for initial baseline
        imgsz=640,              # Standard image size
        batch=16,               # Batch size
        name='bdfd9_defect_detector', # Output folder name
        device='cpu',           # Change to '0' if you have an NVIDIA GPU
        patience=5              # Stop early if no improvement
    )
    
    print("✅ Training complete. Model saved in runs/detect/bdfd9_defect_detector/weights/best.pt")
    
    # Optional: Validate on the test set
    # metrics = model.val()
    # print(f"mAP@0.5: {metrics.box.map50}")

if __name__ == "__main__":
    train_defect_detector()