from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from flask_cors import CORS
import torch
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Carregar modelo PyTorch
model = YOLO('best.pt')

# Definir tamanho de entrada correto para o modelo
INPUT_SIZE = 256

def preprocess_image(image):
    # Redimensionar para o tamanho correto do modelo
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    return img

def process_detections(results):
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                mask = mask.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                class_id = int(result.boxes.cls[i].item())
                confidence = float(result.boxes.conf[i].item())
                
                # Converter máscara para base64
                _, buffer = cv2.imencode('.png', mask)
                mask_base64 = base64.b64encode(buffer).decode('utf-8')
                
                detections.append({
                    'class': 'Moderada' if class_id == 0 else 'Severa',
                    'confidence': confidence,
                    'mask': mask_base64
                })
    
    return detections

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def app_index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Receber imagem em base64
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Converter para numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Falha ao processar imagem'}), 400
        
        # Fazer predição
        results = model.predict(image, conf=0.25)
        
        # Processar detecções
        detections = process_detections(results)
        
        return jsonify({'detections': detections})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()