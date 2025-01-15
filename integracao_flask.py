from flask import Flask, request, jsonify, send_from_directory
import sqlite3
import time
import cv2
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from flask_cors import CORS
from functools import lru_cache
import logging
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# Função para carregar rostos conhecidos do banco de dados
@lru_cache(maxsize=1)
def load_known_faces(page_size=100, page_num=0):
    known_faces = []
    known_names = []
    known_ids = []
    known_cpfs = []
    known_rgs = []
    known_nome_pais = []
    known_nome_maes = []
    known_image_urls = []  # Lista para armazenar os URLs das imagens
    
    # Conectar ao banco de dados SQLite
    conn = sqlite3.connect('meu_banco.db')
    cursor = conn.cursor()

    try:
        # Calcular o offset com base na página e no tamanho da página
        offset = page_num * page_size

        # Obter os dados da tabela users com LIMIT e OFFSET para paginação
        cursor.execute("SELECT id, nome, cpf, rg, nome_pai, nome_mae, urlimagem FROM users LIMIT ? OFFSET ?", (page_size, offset))
        rows = cursor.fetchall()

        for row in rows:
            name = row[1]
            image_url = row[6]
            user_id = row[0]
            cpf = row[2]
            rg = row[3]
            nome_pai = row[4]
            nome_mae = row[5]
            
            # Baixar a imagem usando o URL armazenado no banco de dados
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Erro ao baixar a imagem {image_url}: {e}")
                continue
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = np.array(image)

                # Certificar-se de que a imagem está no formato RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(image_rgb)
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(name)
                    known_ids.append(user_id)
                    known_cpfs.append(cpf)
                    known_rgs.append(rg)
                    known_nome_pais.append(nome_pai)
                    known_nome_maes.append(nome_mae)
                    known_image_urls.append(image_url)  # Adiciona o URL da imagem à lista
    except Exception as e:
        print(f"Erro ao carregar rostos do banco de dados: {e}")
    finally:
        conn.close()

    return known_faces, known_names, known_ids, known_cpfs, known_rgs, known_nome_pais, known_nome_maes, known_image_urls

# Função para comparar a imagem recebida com rostos conhecidos
def compare_image_with_known_faces(image, image_name):
    known_faces, known_names, known_ids, known_cpfs, known_rgs, known_nome_pais, known_nome_maes, known_image_urls  = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    results = []

    # Processa cada rosto detectado
    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.5:
            name = known_names[best_match_index]
            status = "Conhecido"
            user_id = known_ids[best_match_index]
            cpf = known_cpfs[best_match_index]
            rg = known_rgs[best_match_index]
            nome_pai = known_nome_pais[best_match_index]
            nome_mae = known_nome_maes[best_match_index]
            image_url = known_image_urls[best_match_index] 
        else:
            name = "Desconhecido"
            status = "Desconhecido"
            image_url = "" 
        
        results.append({
            "name": name,
            "status": status,
            "cpf": cpf if status == "Conhecido" else "",
            "rg": rg if status == "Conhecido" else "",
            "nome_pai": nome_pai if status == "Conhecido" else "",
            "nome_mae": nome_mae if status == "Conhecido" else "",
            "image_url": image_url  # URL da imagem
        })

    return results

def limpar_cache():
    load_known_faces.cache_clear()
    print("Cache limpo automaticamente!")

scheduler = BackgroundScheduler()
scheduler.add_job(limpar_cache, 'interval', hours=1)  # Limpeza automática a cada 1 hora
scheduler.start()

# Rota para servir imagens (para exibição no frontend)
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('reconhecidos', filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400
    
    if not image_file.content_type.startswith('image/'):
        return jsonify({"error": "O arquivo enviado não é uma imagem válida."}), 400

    try:
        image = Image.open(image_file)
        image = np.array(image)
        results = compare_image_with_known_faces(image, image_file.filename)

        if isinstance(results, list) and "error" in results[0]:
            return jsonify(results), 400
        else:
            return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_faces', methods=['GET'])
def get_faces():
    # Pega os parâmetros de página da requisição (default 0 para página, 100 para size)
    page_num = int(request.args.get('page_num', 0))
    page_size = int(request.args.get('page_size', 100))

    # Chama a função de carregamento de rostos com paginação
    known_faces, known_names, known_ids, known_cpfs, known_rgs, known_nome_pais, known_nome_maes = load_known_faces(page_size, page_num)

    # Retorna os dados paginados em formato JSON
    return jsonify({
        "faces": known_faces,
        "names": known_names,
        "ids": known_ids,
        "cpfs": known_cpfs,
        "rgs": known_rgs,
        "nome_pais": known_nome_pais,
        "nome_maes": known_nome_maes,
    })

@app.route('/refresh_faces', methods=['POST'])
def refresh_faces():
    load_known_faces.cache_clear()
    return jsonify({"message": "Cache atualizado com sucesso."}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
