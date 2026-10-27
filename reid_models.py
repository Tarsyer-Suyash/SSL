import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import base64
import io

# Age group mapping
AGE_IDX_TO_CLASS = {
    0: 'between_10-20',
    1: 'between_20-40',
    2: 'greater_than_40'
}


class ReIDModel:
    """ReID model wrapper for ONNX inference"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def load(self):
        """Load the ONNX model"""
        print(f"Loading ReID ONNX model from: {self.model_path}")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print("ReID ONNX Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading ReID ONNX model: {e}")
            return False

    def extract_features(self, img_pil):
        """Extract ReID features from PIL image"""
        try:
            input_tensor = self._preprocess_image(img_pil)
            input_name = self.session.get_inputs()[0].name
            features = self.session.run(None, {input_name: input_tensor})[0]

            # Normalize features
            norm = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
            features = features / (norm + 1e-10)
            return features.flatten()
        except Exception as e:
            print(f"Warning: Could not extract features. Error: {e}")
            return None

    def _preprocess_image(self, pil_img, height=256, width=128):
        """Preprocess PIL image for ReID model"""
        img = pil_img.resize((width, height), Image.BILINEAR)
        img_data = np.array(img).astype('float32') / 255.0

        norm_mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        norm_std = np.array([0.229, 0.224, 0.225]).astype('float32')

        img_data = (img_data - norm_mean) / norm_std
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)

        return img_data

    def unload(self):
        """Unload the model to free memory"""
        self.session = None
        print("ReID model unloaded.")


class AgeModel:
    """Age prediction model wrapper for ONNX inference"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def load(self):
        """Load the ONNX model"""
        print(f"Loading Age ONNX model from: {self.model_path}")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print("Age ONNX Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading Age ONNX model: {e}")
            return False

    def predict_age(self, img_cv2):
        """Predict age group from OpenCV image"""
        try:
            input_tensor = self._preprocess_image(img_cv2)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})[0]
            pred_class = int(np.argmax(outputs))
            return AGE_IDX_TO_CLASS.get(pred_class, "Unknown")
        except Exception as e:
            print(f"Error predicting age: {e}")
            return "Unknown"

    def _preprocess_image(self, img_cv2):
        """Preprocess OpenCV image for age model"""
        img = cv2.resize(img_cv2, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def unload(self):
        """Unload the model to free memory"""
        self.session = None
        print("Age model unloaded.")


# Utility functions
def decode_image_from_base64(base64_string):
    """Decode base64 string to OpenCV and PIL images"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img_cv2, img_pil
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None, None


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)
