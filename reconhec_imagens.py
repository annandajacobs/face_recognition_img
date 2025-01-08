import os
import cv2
import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Função para carregar rostos conhecidos da API
def load_known_faces():
    known_faces = []
    known_names = []
    # Obter a lista de imagens da API
    images_info = get_images_from_api()

    for image_info in images_info:
        image_url = image_info['image']
        name = image_info['nome']

        # Baixar a imagem da URL fornecida pela API
        image = get_image_from_api(image_url)
        if image is not None:
            # Calcular os encodings do rosto
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)

            if encodings:  # Verifica se encontrou um rosto
                known_faces.append(encodings[0])
                known_names.append(name)
        else:
            print(f"Erro ao obter imagem de {image_url}")
    
    return known_faces, known_names

# Função para obter a lista de imagens e dados da API
def get_images_from_api():
    try:
        response = requests.get('http://127.0.0.1:5000/images/lista')  # Endereço da API para listar imagens
        if response.status_code == 200:
            return response.json()['images']
        else:
            print(f"Erro ao obter lista de imagens. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    return []

# Função para capturar e identificar rostos
def compare_image_with_known_faces(image):
    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return [{"error": "Nenhum rosto conhecido foi carregado."}]  # Retorna lista com um dicionário de erro

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        return [{"error": "Nenhum rosto foi detectado na imagem fornecida."}]

    # Dicionário de resultados
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

        # Salvar o resultado em um dicionário
        results.append({
            "name": name,
            "status": status
        })

        # Desenhar o nome e o bounding box na imagem
        top, right, bottom, left = face_location
        color = (0, 255, 0) if status == "Conhecido" else (0, 0, 255)
        cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
        cv2.putText(rgb_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar ou salvar a imagem resultante
    return results

# Função para carregar a imagem da URL
def get_image_from_api(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return np.array(image)
        else:
            print(f"Erro ao obter imagem de {image_url}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
    return None

# Função para verificar todas as imagens na pasta de upload
def check_images_in_upload_folder(upload_folder):
    # Verifica todas as imagens na pasta 'upload'
    results = []
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            print(f"Verificando imagem: {filename}")
            # Carregar a imagem
            image = cv2.imread(file_path)
            if image is not None:
                result = compare_image_with_known_faces(image)
                results.append({
                    "image": filename,
                    "result": result
                })
            else:
                print(f"Erro ao carregar a imagem {filename}")
    return results

# Execução principal
if __name__ == "__main__":
    # Caminho da pasta de uploads
    upload_folder = 'upload'
    
    # Verifica todas as imagens na pasta de upload
    results = check_images_in_upload_folder(upload_folder)
    
    # Exibe os resultados
    for result in results:
        print(f"Resultados para a imagem {result['image']}:")
        for res in result['result']:
            print(f"Nome: {res['name']}, Status: {res['status']}")
