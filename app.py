from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import onnxruntime as ort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carregar modelo ONNX
onnx_model_path = "best.onnx"  # Nome do seu modelo exportado
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Pegar nomes das entradas e saídas
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Definir tamanho de entrada (ajuste se necessário)
INPUT_SIZE = 640  # Ou 1280, dependendo do seu modelo

# Função para pré-processar imagens
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image.astype(np.float32) / 255.0  # Normaliza para [0,1]
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)
    image = np.expand_dims(image, axis=0)  # Adiciona dimensão batch
    return image

# Função para processar detecções
def process_detections(outputs, conf_threshold=0.5):
    detections = []
    for det in outputs[0]:  # Percorre detecções
        x1, y1, x2, y2, conf, class_id = det[:6]
        if conf > conf_threshold:
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": int(class_id)
            })
    return detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Recebe a imagem em base64
        data = request.json
        image_data = data['image']

        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove o prefixo data:image/jpeg;base64,

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Imagem inválida"}), 400

        # Pré-processa a imagem
        image = preprocess_image(frame)

        # Realiza a inferência com ONNX
        outputs = session.run([output_name], {input_name: image})

        # Processa os resultados
        detections = process_detections(outputs)

        return jsonify({"detections": detections})

    except Exception as e:
        print(f"Erro ao processar a detecção: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)