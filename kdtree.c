#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define EMBEDDING_SIZE 128
#define ID_SIZE 100


typedef struct _face{
    float embedding[EMBEDDING_SIZE];
    char id[ID_SIZE];
}tface;

typedef struct _node {
    void *key; 
    struct _node *esq;
    struct _node *dir;
}tnode;

typedef struct _arv {
    tnode *raiz;
    int k;
    int (*cmp)(void*, void*, int);
    double (*dist)(void*, void*);
}tarv;

typedef struct {
    double dist;
    void *key; 
}heap_item;

typedef struct {
    heap_item *dados;
    int tamanho;
    int capacidade;
}heap;


// ======================= ALOCAÇÃO E COMPARAÇÃO =======================

EXPORT void *aloca_face(float embedding[], char id[]) {
    tface *f = malloc(sizeof(tface));
    if (f == NULL) {
        return NULL;
    }
    memcpy(f->embedding, embedding, sizeof(float) * EMBEDDING_SIZE);
    strncpy(f->id, id, ID_SIZE - 1);
    f->id[ID_SIZE - 1] = '\0';
    return f;
}

int comparador_face(void *a, void *b, int pos) {
    float va = ((tface *)a)->embedding[pos];
    float vb = ((tface *)b)->embedding[pos];
    if(va < vb)
        return -1;
    else if (va > vb)
        return 1;
    else
        return 0;
}

double distancia_face(void *a, void *b) {
    float *ea = ((tface *)a)->embedding;
    float *eb = ((tface *)b)->embedding;
    double soma = 0;
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        double diff = ea[i] - eb[i];
        soma += diff * diff;
    }
    return soma;
}

// ======================= CONSTRUÇÃO DA ÁRVORE =======================

void kdtree_constroi(tarv *arv, int (*cmp)(void *, void *, int), double (*dist)(void *, void *), int k) {
    fprintf(stderr, "DEBUG C: kdtree_constroi: Construindo arvore com k=%d\n", k);
    arv->raiz = NULL;
    arv->cmp = cmp;
    arv->dist = dist;
    arv->k = k;
}

void _kdtree_insere(tnode **raiz, void *key, int (*cmp)(void *, void *, int), int profund, int k) {
    if(*raiz == NULL) {
        *raiz = (tnode*) malloc(sizeof(tnode));
        if (*raiz == NULL) {
            return;
        }
        (*raiz)->key = key;
        (*raiz)->esq = NULL;
        (*raiz)->dir = NULL;
    }
    else{
        int pos = profund % k;
        if(cmp(key, (*raiz)->key, pos) < 0){
            _kdtree_insere(&((*raiz)->esq), key, cmp, profund + 1, k);
        }
        else{
            _kdtree_insere(&((*raiz)->dir), key, cmp, profund + 1, k);
        }
    }
}

EXPORT void kdtree_insere(tarv *arv, void *key) {
    if (key == NULL) {
        return;
    }
    _kdtree_insere(&(arv->raiz), key, arv->cmp, 0, arv->k);
}

// ======================= BUSCA N VIZINHOS COM HEAP =======================

EXPORT void heap_init(heap *h, int capacidade) {
    h->dados = (heap_item*) calloc(capacidade, sizeof(heap_item));
    if (h->dados == NULL) {
        h->capacidade = 0;
        return;
    }
    h->tamanho = 0;
    h->capacidade = capacidade;
}

void heap_insere(heap *h, void *key, double dist) {
    if(key == NULL) {
        return;
    }
    if (h == NULL || h->dados == NULL) {
        return;
    }


    if(h->tamanho < h->capacidade){
        h->dados[h->tamanho].key = key;
        h->dados[h->tamanho].dist = dist;
        h->tamanho++;

        int i = h->tamanho - 1;
        while (i > 0 && h->dados[(i-1)/2].dist < h->dados[i].dist) {
            heap_item tmp = h->dados[(i-1)/2];
            h->dados[(i-1)/2] = h->dados[i];
            h->dados[i] = tmp;
            i = (i-1)/2;
        }
    }
    else if(dist < h->dados[0].dist){
        h->dados[0].key = key;
        h->dados[0].dist = dist;

        int i = 0;
        while (1) {
            int maior = i;
            int esq = 2*i + 1;
            int dir = 2*i + 2;

            if(esq < h->tamanho && h->dados[esq].dist > h->dados[maior].dist)
                maior = esq;
            if(dir < h->tamanho && h->dados[dir].dist > h->dados[maior].dist)
                maior = dir;

            if(maior == i) break;

            heap_item tmp = h->dados[i];
            h->dados[i] = h->dados[maior];
            h->dados[maior] = tmp;
            i = maior;
        }
    } else {
    }
}

void _kdtree_busca_n(tarv *arv, tnode *no, void *query, int profund, heap *h) {
    if (no == NULL) {
        return;
    }

    if (no->key == NULL) {
        return;
    }

    double dist = arv->dist(query, no->key);
    heap_insere(h, no->key, dist);

    int pos = profund % arv->k;
    int comp = arv->cmp(query, no->key, pos); 

    float query_val = ((tface *)query)->embedding[pos];
    float node_val = ((tface *)no->key)->embedding[pos];

    if (comp < 0) {
        _kdtree_busca_n(arv, no->esq, query, profund + 1, h);
    } else {
        _kdtree_busca_n(arv, no->dir, query, profund + 1, h);
    }

    double diff_along_axis = query_val - node_val;
    double dist_to_hyperplane_sq = diff_along_axis * diff_along_axis;

    // Verificar se precisa visitar o outro lado
    if (h->tamanho < h->capacidade || dist_to_hyperplane_sq < h->dados[0].dist) {
        if (comp < 0) {
            _kdtree_busca_n(arv, no->dir, query, profund + 1, h);
        } else {
            _kdtree_busca_n(arv, no->esq, query, profund + 1, h);
        }
    } 
}

EXPORT void kdtree_busca_n_vizinhos(tarv *arv, void *query, int n, heap *h) {
    if (arv->raiz == NULL) {
        fprintf(stderr, "DEBUG C: kdtree_busca_n_vizinhos: Raiz da arvore eh NULL. Nenhum vizinho sera encontrado.\n");
    }
    _kdtree_busca_n(arv, arv->raiz, query, 0, h);
    fprintf(stderr, "DEBUG C: kdtree_busca_n_vizinhos: Busca concluida. Heap tamanho final: %d\n", h->tamanho);
}

// ======================= LIBERAÇÃO DE MEMÓRIA =======================

EXPORT void heap_libera(heap *h) {
    if (h != NULL && h->dados != NULL) {
        free(h->dados);
        h->dados = NULL;
        h->tamanho = 0;
        h->capacidade = 0;
    } 
}

void libera_no(tnode *no) {
    if(no){
        libera_no(no->esq);
        libera_no(no->dir);
        if (no->key) { 
            free(no->key); 
        } 
        free(no);      
    }
}

EXPORT void kdtree_libera(tarv *arv) {
    if (arv) {
        libera_no(arv->raiz);
        arv->raiz = NULL;
    } 
}

// ======================= FUNÇÕES DE INTERFACE GLOBAL (PARA PYTHON) =======================

static tarv *global_kdtree_ptr = NULL;

EXPORT void init_global_kdtree() {
    if (global_kdtree_ptr == NULL) {
        global_kdtree_ptr = (tarv*) malloc(sizeof(tarv));
        if (global_kdtree_ptr == NULL) {
            perror("init_global_kdtree: Erro de alocacao de memoria para global_kdtree_ptr");
            return;
        }
        kdtree_constroi(global_kdtree_ptr, comparador_face, distancia_face, EMBEDDING_SIZE);
    } else {
        fprintf(stderr, "init_global_kdtree: Arvore global ja inicializada.\n");
    }
}

EXPORT tarv* get_global_kdtree() {
    return global_kdtree_ptr;
}

EXPORT void cleanup_global_kdtree() {
    if (global_kdtree_ptr) {
        kdtree_libera(global_kdtree_ptr);
        free(global_kdtree_ptr);
        global_kdtree_ptr = NULL;
    } 
}

EXPORT tface* get_face_from_ptr(void *ptr) {
    return (tface*)ptr;
}

EXPORT void free_ptr(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

int main() {
    return EXIT_SUCCESS;
}