import requests

API_URL = "http://127.0.0.1:8000" 
N_NEIGHBORS = 5 

QUERY_EMBEDDING = []


def query_kdtree_with_embedding():
    if len(QUERY_EMBEDDING) != 128:
        print(f"ERRO: O embedding de consulta deve ter 128 floats, mas possui {len(QUERY_EMBEDDING)}.")
        return

    print("Enviando embedding de consulta para a API...")
    endpoint_query = f"{API_URL}/reconhecer_face?n_vizinhos={N_NEIGHBORS}"
    data = {
        "query_embedding": QUERY_EMBEDDING
    }
    try:
        response = requests.post(endpoint_query, json=data)
        response.raise_for_status()
        results = response.json()

        print("\n--- Resultados da Busca dos Vizinhos Mais Próximos ---")
        if results and "vizinhos" in results and results["vizinhos"]:
            for i, vizinho in enumerate(results["vizinhos"]):
                print(f"{i+1}. ID: {vizinho['id']}, Distância: {vizinho['distancia']:.4f}")

    except requests.exceptions.RequestException as e:
        print(f"ERRO ao consultar a API: {e}")
        if response is not None:
            print(f"Resposta da API: {response.status_code} - {response.text}")

if __name__ == "__main__":
    query_kdtree_with_embedding()