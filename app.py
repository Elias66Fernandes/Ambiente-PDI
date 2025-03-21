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
    # Verifica se a string base64 tem o prefixo data:image
    if 'data:image' in image_data:
        # Decodifica a imagem base64
        image_data = image_data.split(',')[1]
    # Decodifica os dados base64 em bytes
    image_bytes = base64.b64decode(image_data)
    # Abre a imagem usando PIL
    image = Image.open(BytesIO(image_bytes))
    # Redimensiona para 256x256 se necessário
    if image.size != (256, 256):
        image = image.resize((256, 256))
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
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("Dados da imagem não fornecidos")
            
        # Extrai a string base64 da imagem
        image_data = data['image']
        if isinstance(image_data, dict):
            image_data = image_data.get('dataUrl', '')
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
        results = model.predict(image, conf=0.55, verbose=False)
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
                    mask = mask.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
                    mask = np.array(mask)
                    
                    # Aplica um pequeno blur para suavizar bordas
                    import cv2
                    mask = cv2.GaussianBlur(mask, (3,3), 0)
                    
                    # Encontra os contornos com mais detalhes
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                    
                    if contours:
                        # Pega o maior contorno
                        contour = max(contours, key=cv2.contourArea)
                        
                        # Suaviza o contorno
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Aplica spline para suavizar ainda mais
                        if len(approx) > 2:
                            # Converte para array numpy
                            curve = approx.squeeze()
                            
                            # Fecha o contorno
                            curve = np.vstack((curve, curve[0]))
                            
                            # Gera pontos interpolados
                            t = np.arange(len(curve))
                            ti = np.linspace(0, len(curve)-1, len(curve)*2)
                            
                            # Interpola x e y separadamente
                            x = np.interp(ti, t, curve[:, 0])
                            y = np.interp(ti, t, curve[:, 1])
                            
                            # Suaviza os pontos interpolados
                            sigma = 1
                            x = cv2.GaussianBlur(x.reshape(-1, 1), (1, 5), sigma).squeeze()
                            y = cv2.GaussianBlur(y.reshape(-1, 1), (1, 5), sigma).squeeze()
                            
                            # Limita os pontos à bounding box
                            x = np.clip(x, x1, x2)
                            y = np.clip(y, y1, y2)
                            
                            # Converte para lista de pontos
                            for i in range(len(x)):
                                mask_points.append([float(x[i]), float(y[i])])
                        else:
                            # Se tiver poucos pontos, usa o contorno original
                            for point in approx:
                                x, y = point[0]
                                # Limita os pontos à bounding box
                                x = np.clip(x, x1, x2)
                                y = np.clip(y, y1, y2)
                                mask_points.append([float(x), float(y)])
                except Exception as e:
                    print(f"Erro ao processar máscara: {str(e)}")
                    mask_points = []
                
                # Adiciona o tipo da fissura
                if cls == 0:
                    fissure_type = "Moderada"
                elif cls == 1:
                    fissure_type = "Severa"
                else:
                    fissure_type = "Negativa"
                
                final_detections.append({
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'confidence': float(conf),
                    'class': int(cls),
                    'mask_points': mask_points,
                    'fissure_type': fissure_type
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