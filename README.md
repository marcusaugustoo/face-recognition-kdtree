# Face Recognition API — KD-Tree & KNN

> Sistema de reconhecimento facial de utilizando **C** para estruturas de dados e **Python** para a API e interface.

---

## 📖 Visão Geral

Este projeto implementa uma **KD-Tree** otimizada para realizar buscas em vetores de *embeddings* faciais de **128 dimensões**, permitindo encontrar rapidamente os vizinhos mais próximos.

Também foi implementado um mecanismo de **KNN** utilizando **Max-Heap**, garantindo consultas de múltiplos vizinhos sem perda de eficiência.

O objetivo do trabalho foi refatorar uma base inicial, melhorando modularidade, desempenho e escalabilidade.

---

## Funcionalidades

### KD-Tree para embeddings de 128 dimensões
- Suporte total a vetores de 128 floats.
- Armazena identificadores (strings) associados a cada face.
- Estruturada para dados de alta dimensionalidade usados em biometria real.

### Busca KNN com Heap
- Implementação de **Max-Heap** para armazenar candidatos durante a busca.
- Backtracking inteligente com poda.
- Retorno eficiente dos **N vizinhos mais próximos**.

### API em Python com FastAPI
- Endpoints rápidos e simples.
- Integração direta com o módulo C.

---

## 🗂 Dataset Utilizado

As faces utilizadas para testes são provenientes do dataset **LFW – Labeled Faces in the Wild**, amplamente usado em pesquisa.

- **Fonte:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/atulanandjha/lfwpeople  
- **Quantidade utilizada:** ~1000 embeddings faciais.

---

## Estrutura do Projeto

```plaintext
/
├── auxiliar/                    # Scripts auxiliares e dados do projeto
│   ├── embeddings/              # Vetores de embeddings utilizados na KD-Tree
│   ├── popular.py               # Script para popular a árvore com embeddings
│   └── reconhecer_face.py       # Script para realizar a busca KNN e reconhecer uma face
│
├── app.py                       # Servidor da API (FastAPI)
├── kdtree.c                     # Implementação da KD-Tree + Max-Heap em C
├── kdtree_wrapper.py            # Interface entre Python e o módulo em C
└── README.md                    # Documento do projeto

```

---

## Como Executar

### 🔧 Pré-requisitos
- GCC (ou outro compilador C)
- Python 3.8+
- pip

---

## Passo 1 — Compilar o módulo C

### **Linux / macOS**
```bash
gcc -shared -o kdtree.so -fPIC kdtree.c
```

### **Windows**
```bash
gcc -shared -o kdtree.dll kdtree.c
```

---

## Passo 2 — Instalar dependências Python

```bash
pip install fastapi uvicorn numpy pydantic
```

---

## Passo 3 — Rodar a API

```bash
uvicorn app:app --reload
```

A API ficará disponível em:

```
http://127.0.0.1:8000
```

---

## Melhorias Futuras 

- Implementar balanceamento automático da KD-Tree.
- Adicionar cache LRU para resultados de consultas repetidas.
- Criar interface web minimalista.

---

