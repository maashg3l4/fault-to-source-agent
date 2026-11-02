import yaml
import json
import os
from ultralytics import YOLO

# --- 1. Load the Mapping ---
# Ensure this file exists: config/vision_to_causal_map.yaml
config_path = 'config/vision_to_causal_map.yaml'
if not os.path.exists(config_path):
    # Fallback if the folder wasn't created
    os.makedirs('config', exist_ok=True)
    default_mapping = {
        'DropStitch': {'causal_variable': 'VisualMicrostructure', 'causal_state': 'Degraded'},
        'Hole': {'causal_variable': 'VisualMicrostructure', 'causal_state': 'Degraded'},
        'Oil Spot': {'causal_variable': 'DyeLotQuality', 'causal_state': 'Bad'},
        'Knot': {'causal_variable': 'VisualMicrostructure', 'causal_state': 'Degraded'},
        'Yarn Dust': {'causal_variable': 'YarnQuality', 'causal_state': 'Poor'},
        'Default': {'causal_variable': 'VisualMicrostructure', 'causal_state': 'Degraded'}
    }
    with open(config_path, 'w') as f:
        yaml.dump(default_mapping, f)
    mapping = default_mapping
else:
    with open(config_path, 'r') as f:
        mapping = yaml.safe_load(f)

# --- 2. Load the Local Model ---
print("Loading YOLOv8 model...")
# Ensure best.pt is in the models folder
model = YOLO('models/best.pt')

def image_to_causal_evidence(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    print(f"🔍 Running inference on {image_path}...")
    results = model(image_path)
    
    evidence = {}
    highest_conf = 0.0
    detected_class = None

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > highest_conf:
                highest_conf = conf
                detected_class = model.names[int(box.cls[0])]

    if detected_class:
        print(f"✅ Detected: {detected_class} (Confidence: {highest_conf:.2f})")
        
        # Map to causal variable
        mapping_entry = mapping.get(detected_class, mapping.get('Default'))
        causal_var = mapping_entry['causal_variable']
        causal_state = mapping_entry['causal_state']
        
        evidence[causal_var] = causal_state
        
        # This JSON is what your LangGraph Agent will consume
        output_payload = {
            "defect_type": detected_class,
            "confidence": highest_conf,
            "causal_evidence": evidence
        }
        return output_payload
    else:
        print("No defects detected.")
        return None

if __name__ == "__main__":
    # The exact path you found
    test_image = "data/raw/BDFD9 Bangladesh Centric Fabric Defect Dataset/BDFD9/test/DropStitch/images/IMG_1054_webp.rf.4oN5zFCFPSSggJVcdytK.webp"
    
    payload = image_to_causal_evidence(test_image)
    
    if payload:
        print("\n🚀 Modality Bridge Output (JSON Payload for Causal Engine):")
        print(json.dumps(payload, indent=2))