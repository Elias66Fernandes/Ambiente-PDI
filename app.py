from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__, static_folder='static')
CORS(app)

# Carrega o modelo YOLOv8
try:
    model_path = os.path.join('static', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    model = YOLO(model_path)
    print("Modelo YOLOv11 carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {str(e)}")
    model = None

def preprocess_image(image_data):
    # Decodifica a imagem base64
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    return image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def app_index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        print("Iniciando detecção...")
        
        # Recebe a imagem do frontend
        data = request.json
        if not data or 'image' not in data:
            raise ValueError("Dados da imagem não fornecidos")
        image_data = data['image']
        print("Imagem recebida com sucesso")
        
        # Verifica se o modelo foi carregado
        if model is None:
            raise RuntimeError("Modelo não foi carregado corretamente")
        
        # Pré-processa a imagem
        print("Iniciando pré-processamento...")
        image = preprocess_image(image_data)
        print("Pré-processamento concluído")
        
        # Executa a inferência
        print("Iniciando inferência...")
        results = model.predict(image, conf=0.25, verbose=False)
        print("Inferência concluída")
        
        # Processa as detecções
        print("Processando detecções...")
        final_detections = []
        
        # Pega o primeiro resultado (primeira imagem)
        result = results[0]
        
        # Obtém as dimensões originais da imagem
        orig_width, orig_height = image.size
        
        if result.masks is not None:
            # Converte as máscaras para o formato da imagem original
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Extrai as coordenadas da box e a confiança
                x1, y1, x2, y2, conf, cls = box
                
                # Normaliza as coordenadas para a escala da imagem original
                x1, x2 = x1.item(), x2.item()
                y1, y2 = y1.item(), y2.item()
                
                # Processa a máscara
                mask_points = []
                try:
                    # Encontra os contornos da máscara
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask = Image.fromarray(mask)
                    mask = mask.resize((orig_width, orig_height))
                    mask = np.array(mask)
                    
                    # Converte a máscara em uma lista de pontos do contorno
                    import cv2
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Pega o maior contorno
                        contour = max(contours, key=cv2.contourArea)
                        # Simplifica o contorno para reduzir o número de pontos
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Converte para lista de pontos
                        for point in approx:
                            x, y = point[0]
                            mask_points.append([float(x), float(y)])
                except Exception as e:
                    print(f"Erro ao processar máscara: {str(e)}")
                    mask_points = []
                
                final_detections.append({
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'confidence': float(conf),
                    'class': int(cls),
                    'mask_points': mask_points
                })
        
        print(f"Encontradas {len(final_detections)} detecções")
        print("Detecção finalizada com sucesso")
        return jsonify({'detections': final_detections})
        
    except Exception as e:
        print('Erro na detecção:', str(e))
        import traceback
        print('Traceback:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)