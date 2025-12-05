import requests
import os
import time 

EMBEDDINGS_FILE = "embeddings.txt"
API_URL = "http://127.0.0.1:8000" 
def populate_kdtree_from_file():
    print(f"Verificando arquivo de embeddings: {EMBEDDINGS_FILE}")

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"--- POPULAR.PY FINALIZADO COM ERRO ---")
        return

    print(f"Arquivo '{EMBEDDINGS_FILE}' encontrado. Iniciando conexão com a API em {API_URL}")
    print(f"Conexão com a API estabelecida. Começando a ler e enviar faces...")

    processed_count = 0
    error_count = 0
    with open(EMBEDDINGS_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100 == 0: 
                print(f"Progresso: Processando linha {line_num}...")

            parts = line.strip().split()
            if not parts: 
                continue

            face_id = parts[0]
            try:
                embedding = [float(x) for x in parts[1:]]
            except ValueError:
                error_count += 1
                continue

            if len(embedding) != 128:
                error_count += 1
                continue

            endpoint_add = f"{API_URL}/adicionar_face"
            data = {
                "embedding": embedding,
                "id": face_id
            }
            try:
                response = requests.post(endpoint_add, json=data, timeout=5) # Adiciona timeout
                response.raise_for_status()
                processed_count += 1
            except requests.exceptions.Timeout:
                print(f"ERRO DE TIMEOUT: Requisição para adicionar '{face_id}' (Linha {line_num}) excedeu o tempo limite.")
                error_count += 1
            except requests.exceptions.RequestException as e:
                print(f"ERRO NA REQUISIÇÃO POST para '{face_id}' (Linha {line_num}): {e}")
                if response is not None:
                    print(f"Resposta da API: Status {response.status_code} - {response.text}")
                error_count += 1


    print(f"Total de faces tentadas: {line_num}")
    print(f"Faces enviadas com sucesso: {processed_count}")
    print(f"Faces com erro: {error_count}")
    print(f"--- POPULAR.PY FINALIZADO ---")

if __name__ == "__main__":
    populate_kdtree_from_file()