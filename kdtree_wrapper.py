import ctypes
from ctypes import Structure, POINTER, c_double, c_int, c_char, c_float, CFUNCTYPE, c_void_p
import os 
EMBEDDING_SIZE = 128
ID_SIZE = 100

class TFace(Structure):
    _fields_ = [
        ("embedding", c_float * EMBEDDING_SIZE),  
        ("id", c_char * ID_SIZE)                  
    ]

class HeapItem(Structure):
    _fields_ = [
        ("dist", c_double),
        ("key", POINTER(TFace))  
    ]

class Heap(Structure):
    _fields_ = [
        ("dados", POINTER(HeapItem)),
        ("tamanho", c_int),
        ("capacidade", c_int)
    ]

class TNode(Structure):
    pass 

class Tarv(Structure):
    _fields_ = [
        ("raiz", POINTER(TNode)),
        ("k", c_int),
        ("cmp", CFUNCTYPE(c_int, c_void_p, c_void_p, c_int)),
        ("dist", CFUNCTYPE(c_double, c_void_p, c_void_p))
    ]

TNode._fields_ = [
    ("key", c_void_p), 
    ("esq", POINTER(TNode)),
    ("dir", POINTER(TNode))
]

library_path = os.path.join(os.getcwd(), "libkdtree.so")
lib = ctypes.CDLL(library_path)

lib.aloca_face.argtypes = [
    c_float * EMBEDDING_SIZE,  
    c_char * ID_SIZE           
]
lib.aloca_face.restype = c_void_p 

lib.init_global_kdtree.argtypes = []
lib.init_global_kdtree.restype = None

lib.get_global_kdtree.argtypes = []
lib.get_global_kdtree.restype = POINTER(Tarv)

lib.cleanup_global_kdtree.argtypes = []
lib.cleanup_global_kdtree.restype = None

lib.kdtree_insere.argtypes = [POINTER(Tarv), c_void_p]
lib.kdtree_insere.restype = None

lib.kdtree_busca_n_vizinhos.argtypes = [POINTER(Tarv), c_void_p, c_int, POINTER(Heap)]
lib.kdtree_busca_n_vizinhos.restype = None 

lib.kdtree_libera.argtypes = [POINTER(Tarv)]
lib.kdtree_libera.restype = None

lib.heap_init.argtypes = [POINTER(Heap), c_int]
lib.heap_init.restype = None

lib.heap_libera.argtypes = [POINTER(Heap)]
lib.heap_libera.restype = None

lib.get_face_from_ptr.argtypes = [c_void_p]
lib.get_face_from_ptr.restype = POINTER(TFace)

try:
    lib.free_ptr.argtypes = [c_void_p]
    lib.free_ptr.restype = None
except AttributeError:
    pass