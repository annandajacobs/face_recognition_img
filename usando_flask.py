from flask import Flask, jsonify
import os
import json

app = Flask(__name__)

# Dados falsos das imagens
image_data = {
    1: {
       "imagem": "https://raw.githubusercontent.com/annandajacobs/face_recognition_img/refs/heads/luana/images/pessoa.jpeg",
       "nome": "Gustavo",
    },
    2: {
        "imagem": "https://raw.githubusercontent.com/annandajacobs/face_recognition_img/refs/heads/luana/images/pessoa2.jpg",
       "nome": "annanda",
    },
    3: {
        "imagem": "https://raw.githubusercontent.com/annandajacobs/face_recognition_img/refs/heads/luana/images/pessoa3.jpg",
       "nome": "Ana",
    },
    4:{
        "imagem":"https://raw.githubusercontent.com/annandajacobs/face_recognition_img/refs/heads/luana/images/pessoa4.jpg",
        "nome":"Robert",
    }
}

# Função para carregar os dados falsos
def load_image_data():
    return image_data

# Rota para listar todas as imagens e seus dados falsos
@app.route('/images/lista', methods=['GET'])
def get_image_list():
    try:
        # Carregar dados falsos
        image_data = load_image_data()

        # Preparar a resposta com os dados falsos
        images_info = []
        for image_id, data in image_data.items():
            images_info.append({
                "image": data["imagem"],
                "nome": data["nome"]
            })
        
        print(images_info)  # Log para verificar os dados retornados
        return jsonify({"images": images_info})
    except Exception as e:
        return jsonify({"error": f"Erro ao listar imagens: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
