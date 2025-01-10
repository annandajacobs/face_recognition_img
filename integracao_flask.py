from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa o CORS
import sqlite3
import os
import cv2
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)
CORS(app)  # Habilita o CORS para todo o aplicativo

# Função para carregar rostos conhecidos do banco de dados
def load_known_faces():
    known_faces = []
    known_names = []
    conn = sqlite3.connect('meu_banco.db')
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id, nome, urlimagem FROM users")
        rows = cursor.fetchall()

        for row in rows:
            name = row[1]
            image_url = row[2]

            # Baixar a imagem usando o URL armazenado no banco de dados
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = np.array(image)

                # Certificar-se de que a imagem está no formato RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(image_rgb)
                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(name)
    except Exception as e:
        print(f"Erro ao carregar rostos do banco de dados: {e}")
    finally:
        conn.close()

    return known_faces, known_names

# Função para comparar a imagem recebida com rostos conhecidos
def compare_image_with_known_faces(image):
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    results = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.5:
            name = known_names[best_match_index]
            status = "Conhecido"
        else:
            name = "Desconhecido"
            status = "Desconhecido"

        results.append({"name": name, "status": status})

    return results

# Rota para receber o upload de imagem
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400

    try:
        image = Image.open(image_file)
        image = np.array(image)
        results = compare_image_with_known_faces(image)

        if isinstance(results, list) and "error" in results[0]:
            return jsonify(results), 400
        else:
            return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)