# 🎭 Face Recognition API (KD-Tree & KNN)

> Sistema de reconhecimento facial de alta performance utilizando **C** para estruturas de dados (KD-Tree + Heap) e **Python** (FastAPI) para a interface.

![Language](https://img.shields.io/badge/language-C-blue)
![Language](https://img.shields.io/badge/language-Python-yellow)
![Framework](https://img.shields.io/badge/framework-FastAPI-green)

## 📖 Descrição do Projeto

Este trabalho consiste no desenvolvimento de um sistema de busca de reconhecimento facial otimizado. O núcleo do projeto é uma **KD-Tree (K-Dimensional Tree)**, uma estrutura de dados de partição de espaço binário, utilizada aqui para indexar e buscar embeddings faciais de 128 dimensões.

O objetivo principal foi refatorar uma implementação base para suportar buscas eficientes dos **N vizinhos mais próximos** (KNN), utilizando uma estrutura de **Heap** para priorização.

### ✨ Diferenciais Implementados

1.  **Refatoração para 128 Dimensões:**
    * Adaptação da estrutura de dados para suportar vetores de *embeddings* (128 floats) e identificadores de usuários (strings), simulando um cenário real de biometria facial.

2.  **Busca KNN com Heap:**
    * Implementação de um **Max-Heap** para gerenciar os candidatos a vizinhos mais próximos durante a navegação na árvore.
    * Isso permite retornar não apenas o vizinho mais próximo, mas os **N** mais similares, com poda eficiente da árvore (backtracking otimizado).

### Fonte dos Dados
As faces utilizadas para povoar a base de dados foram retiradas do dataset público **LFW (Labeled Faces in the Wild)**, disponível no Kaggle.
* **Dataset:** [LFW - People (Face Recognition)](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)
* **Quantidade:** Foram inseridos vetores de características (embeddings) de aproximadamente **1000 faces** distintas na árvore.

## 📂 Estrutura dos Arquivos

* `kdtree.c`: Código fonte em C contendo a implementação da KD-Tree, do Heap e das funções de distância euclidiana.
* `app.py`: Servidor da API construído com FastAPI.
* `kdtree_wrapper.py`: Interface de ligação entre Python e C.

## 🚀 Instalação e Execução

### Pré-requisitos
* GCC (ou outro compilador C)
* Python 3.8+

### Passo 1: Compilar a Biblioteca C
A API Python precisa carregar o código C compilado como uma biblioteca dinâmica (`.so`).

**No Linux/MacOS:**
```bash
gcc -shared -o kdtree.so -fPIC kdtree.c


**No Windows:**
```bash
gcc -shared -o kdtree.dll kdtree.c

### Passo 2: Instalar Dependências do Python
```bash
pip install fastapi uvicorn numpy pydantic

### Passo 3: Rodar a API
```bash
uvicorn app:app --reload