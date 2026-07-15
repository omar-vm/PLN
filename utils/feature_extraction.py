from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


def extraer_caracteristicas_bow(corpus: list) -> tuple:
    vectorizer = CountVectorizer()
    matriz_bow = vectorizer.fit_transform(corpus)
    vocabulario = vectorizer.get_feature_names_out()

    return matriz_bow, vocabulario

def extraer_caracteristicas_tfidf(corpus: list) -> tuple:
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(corpus)
    vocabulario = vectorizer.get_feature_names_out()

    return matriz_tfidf, vocabulario




def procesar_corpus_oraciones(nlp, texto: str) -> list:
    """Tokeniza, lematiza y limpia el texto, retornando una lista de oraciones."""
    doc = nlp(texto)
    sentences = []
    
    for sent in doc.sents:
        tokens = [
            token.lemma_.lower() 
            for token in sent 
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        if len(tokens) > 1:
            sentences.append(tokens)
            
    return sentences

def entrenar_modelo_word2vec(oraciones: list, dimensiones: int = 10) -> Word2Vec:
    """Entrena y retorna un modelo Word2Vec a partir de oraciones procesadas."""
    modelo = Word2Vec(
        oraciones,
        vector_size=dimensiones,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count(),
        seed=42 
    )
    return modelo

def aplicar_pca_embeddings(modelo: Word2Vec, n_componentes: int = 3) -> pd.DataFrame:
    """Reduce la dimensionalidad de los embeddings para su visualización."""
    vocabulario = list(modelo.wv.index_to_key)
    vectores = modelo.wv[vocabulario]
    
    pca = PCA(n_components=n_componentes)
    vectores_reducidos = pca.fit_transform(vectores)
    
    # Empaquetar en DataFrame para facilitar el manejo en la vista
    df = pd.DataFrame(vectores_reducidos, columns=['x', 'y', 'z'])
    df['palabra'] = vocabulario
    return df


def obtener_vecindario_pca(modelo, df_pca: pd.DataFrame, palabra_objetivo: str, top_n: int = 15) -> pd.DataFrame:
    """
    Filtra el DataFrame de PCA para dejar solo una palabra y sus vecinos semánticos.
    """
    if palabra_objetivo not in modelo.wv:
        print(f"La palabra '{palabra_objetivo}' no está en el vocabulario.")
        return pd.DataFrame()

    # Obtener palabras similares usando el modelo Word2Vec original
    similares = modelo.wv.most_similar(palabra_objetivo, topn=top_n)
    
    # Crear lista con la palabra central + sus vecinos
    palabras_cercanas = [palabra_objetivo] + [p[0] for p in similares]

    # Filtrar el DataFrame de PCA
    df_filtrado = df_pca[df_pca['palabra'].isin(palabras_cercanas)].copy()
    return df_filtrado