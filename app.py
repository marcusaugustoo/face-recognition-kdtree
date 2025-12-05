from fastapi import FastAPI, Query, HTTPException
from kdtree_wrapper import lib, TFace, Heap, EMBEDDING_SIZE, ID_SIZE
from ctypes import cast, c_float, c_char, POINTER, create_string_buffer, CDLL, c_void_p
from pydantic import BaseModel, Field
import numpy as np
import ctypes 

app = FastAPI()

class FaceEntrada(BaseModel):
    embedding: list[float] = Field(..., min_length=EMBEDDING_SIZE, max_length=EMBEDDING_SIZE)
    id: str = Field(..., max_length=ID_SIZE - 1)

class VizinhoProximo(BaseModel):
    id: str
    distancia: float

class QueryFaceRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=EMBEDDING_SIZE, max_length=EMBEDDING_SIZE)

# Rotas da API
@app.on_event("startup")
async def startup_event():
    lib.init_global_kdtree()
    print("KDTree global inicializada.")

@app.on_event("shutdown")
async def shutdown_event():
    arv = lib.get_global_kdtree()
    if arv:
        lib.kdtree_libera(arv)
        lib.cleanup_global_kdtree()
        print("KDTree global liberada.")

@app.post("/adicionar_face")
def adicionar_face(face: FaceEntrada):
    if len(face.embedding) != EMBEDDING_SIZE:
        raise HTTPException(status_code=400, detail=f"Embedding deve ter {EMBEDDING_SIZE} floats.")

    c_embedding_array = (c_float * EMBEDDING_SIZE)(*face.embedding)
    id_bytes = face.id.encode('utf-8')
    if len(id_bytes) >= ID_SIZE:
        id_bytes = id_bytes[:ID_SIZE - 1]
    c_id_buffer = create_string_buffer(id_bytes, ID_SIZE)

    face_ptr = lib.aloca_face(c_embedding_array, c_id_buffer)
    if not face_ptr:
        raise HTTPException(status_code=500, detail="Falha ao alocar memória para a face em C.")

    arv = lib.get_global_kdtree()
    if not arv:
        raise HTTPException(status_code=500, detail="KDTree não inicializada.")

    lib.kdtree_insere(arv, face_ptr)
    return {"mensagem": f"Face '{face.id}' adicionada com sucesso."}

@app.post("/reconhecer_face")
def reconhecer_face(query_data: QueryFaceRequest, n_vizinhos: int = Query(1, gt=0)):

    query_embedding = query_data.query_embedding

    if len(query_embedding) != EMBEDDING_SIZE:
        raise HTTPException(status_code=400, detail=f"Embedding de consulta deve ter {EMBEDDING_SIZE} floats.")

    arv = lib.get_global_kdtree()
    if not arv:
        raise HTTPException(status_code=500, detail="KDTree não inicializada.")

    c_query_embedding_array = (c_float * EMBEDDING_SIZE)(*query_embedding)
    temp_id_buffer = create_string_buffer(b'', ID_SIZE)
    query_face_ptr = lib.aloca_face(c_query_embedding_array, temp_id_buffer)

    if not query_face_ptr:
        print(f"Aloca_face retornou NULL para query_face_ptr.")
        raise HTTPException(status_code=500, detail="Falha ao alocar memória para a consulta em C.")
    else:
        print(f"Aloca_face retornou query_face_ptr = {query_face_ptr}")

    python_heap = Heap()
    lib.heap_init(POINTER(Heap)(python_heap), n_vizinhos)

    lib.kdtree_busca_n_vizinhos(arv, query_face_ptr, n_vizinhos, POINTER(Heap)(python_heap))
    print(f"KDtree_busca_n_vizinhos retornou. Heap tamanho: {python_heap.tamanho}")

    resultados = []

    for i in range(python_heap.tamanho):
        item = python_heap.dados[i]
        face_data_ptr = lib.get_face_from_ptr(item.key)
        if face_data_ptr and face_data_ptr.contents.id:
            try:
                person_id = face_data_ptr.contents.id.decode('utf-8').strip('\0')
                resultados.append(VizinhoProximo(id=person_id, distancia=item.dist))
            except UnicodeDecodeError:
                print(f"Erro ao decodificar ID para o item na posição {i}.")
                continue
        else:
            print(f"Aviso: item.key ou face_data_ptr.contents.id é nulo na posição {i}.")

    try:
        msvcrt = ctypes.CDLL('msvcrt')
        msvcrt.free.argtypes = [ctypes.c_void_p]
        if query_face_ptr: 
            msvcrt.free(query_face_ptr)
        else:
            print(f"Query_face_ptr já era NULL, não precisou liberar.")
    except Exception as e:
        print(f"Aviso: Não foi possível liberar query_face_ptr usando msvcrt.free: {e}")

    lib.heap_libera(POINTER(Heap)(python_heap))

    resultados.sort(key=lambda x: x.distancia)

    return {"vizinhos": resultados}