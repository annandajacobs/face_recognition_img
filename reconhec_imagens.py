import sqlite3
import os
import cv2
import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Função para carregar rostos conhecidos do banco de dados
def load_known_faces():
    known_faces = []
    known_names = []
    # Conectar ao banco de dados SQLite
    conn = sqlite3.connect('meu_banco.db')  # Substitua com o caminho correto do seu banco de dados
    cursor = conn.cursor()

    try:
        # Obter os dados da tabela users
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
                
                # Obter os "encodings" do rosto
                encodings = face_recognition.face_encodings(image_rgb)
                if encodings:  # Verifica se encontrou um rosto
                    known_faces.append(encodings[0])
                    known_names.append(name)  # Armazenar apenas o nome do conhecido
    except Exception as e:
        print(f"Erro ao carregar rostos do banco de dados: {e}")
    finally:
        conn.close()

    return known_faces, known_names

# Função para capturar e identificar rostos a partir da imagem
def compare_image_with_known_faces(image, image_name):
    # Carregar os rostos conhecidos
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    results = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Comparação com rostos conhecidos
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.5:  # Limite ajustável
            name = known_names[best_match_index]
            status = "Conhecido"
        else:
            name = "Desconhecido"
            status = "Desconhecido"

        # URL da imagem que está na API (mesmo para conhecido ou desconhecido)
        image_url = f"http://localhost:5000/images/{image_name}"

        # Salvar o resultado em um dicionário
        results.append({
            "name": name,
            "status": status,
            "image_url": image_url  # A URL da API será retornada em ambos os casos
        })

    return results

if __name__ == "__main__":
    # Fazer uma requisição à API para pegar o image_url
    api_url = 'http://localhost:5000/images/lista'  # URL da sua API Flask
    response = requests.get(api_url)

    if response.status_code == 200:
        # Supondo que a API retorne uma lista de imagens e seus dados
        images_data = response.json().get("images", [])
        
        # Aqui você pode iterar sobre a lista de imagens e comparar uma por uma
        for image_data in images_data:
            image_name = image_data.get('image')  # Nome da imagem
            image_url = f"http://localhost:5000/images/{image_name}"  # URL completa da imagem para a API
            
            # Fazer o download da imagem
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))
                image = np.array(image)
                result = compare_image_with_known_faces(image, image_name)
                
                if isinstance(result, list) and "error" in result[0]:
                    print(result[0]["error"])
                else:
                    # Exibir os resultados de forma simplificada
                    for res in result:
                        print(f"Nome: {res['name']}, Status: {res['status']}, URL da Imagem: {res['image_url']}")
            else:
                print(f"Erro ao baixar a imagem de {image_url}.")
    else:
        print("Erro ao acessar a lista de imagens na API.")
