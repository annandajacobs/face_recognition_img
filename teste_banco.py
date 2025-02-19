import requests

url = 'https://raw.githubusercontent.com/annandajacobs/face_recognition_img/refs/heads/main/images/gustavo_1.jpg'
response = requests.get(url)

if response.status_code == 200:
    print("Imagem acessada com sucesso!")
else:
    print(f"Erro ao acessar a imagem. Status code: {response.status_code}")
