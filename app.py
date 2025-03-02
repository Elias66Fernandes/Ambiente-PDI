from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
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
    print("Modelo YOLOv8 carregado com sucesso!")
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
        results = model.predict(image, conf=0.25)  # Ajuste o conf threshold conforme necessário
        print("Inferência concluída")
        
        # Processa as detecções
        print("Processando detecções...")
        final_detections = []
        
        # Pega o primeiro resultado (primeira imagem)
        result = results[0]
        
        # Para segmentação, vamos pegar as máscaras e boxes
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                # Converte a máscara para o formato da imagem original
                mask = mask.cpu().numpy()
                
                # Pega a bounding box correspondente
                box = result.boxes.data[i]
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                
                final_detections.append({
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'confidence': float(conf),
                    'class': int(cls),
                    'mask': mask.tolist()  # Converte a máscara para lista para serialização JSON
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