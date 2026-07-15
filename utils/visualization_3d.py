from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # Import necesario para 3D
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# FUNCIÓN AUXILIAR PARA GRAFICAR EN 3D
# ---------------------------------------------------------
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos):
    # 1. TRANSPONER: Filas = Palabras, Columnas = Contextos
    matriz_palabras = matriz.T
    
    # 2. PCA: Reducir a 3 DIMENSIONES
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())
    
    # Extraer coordenadas X, Y, Z
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    # 3. GRAFICAR SCATTER 3D
    # Usamos profundidad visual (depthshade=True) para ayudar a la perspectiva
    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors='k', alpha=0.8, depthshade=True)
    
    # Etiquetar puntos
    for i, palabra in enumerate(vocabulario):
        # Agregamos un pequeño offset a Z para que el texto flote sobre el punto
        ax.text(x[i], y[i], z[i] + 0.1, palabra, fontsize=9)
        
    ax.set_title(titulo)
    ax.set_xlabel('Comp. Principal 1')
    ax.set_ylabel('Comp. Principal 2')
    ax.set_zlabel('Comp. Principal 3')
    
    # Líneas de referencia en el origen (0,0,0)
    ax.plot([0,0], [0,0], [z.min(), z.max()], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([x.min(), x.max()], [0,0], [0,0], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([0,0], [y.min(), y.max()], [0,0], c='grey', ls='--', lw=0.5, alpha=0.3)


import pandas as pd

def graficar_espacio_semantico_3d(ax, df_3d: pd.DataFrame, titulo: str) -> None:
    """
    Renderiza un Scatter Plot 3D con etiquetas de texto basándose en 
    coordenadas extraídas de Embeddings.
    """
    # Renderizar puntos
    ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'], 
               c='crimson', s=80, edgecolors='white', alpha=0.8)

    # Colocar etiquetas de palabras
    for i, row in df_3d.iterrows():
        ax.text(row['x'], row['y'], row['z'], f" {row['palabra']}", size=10)

    # Configuración de estilo
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Dimensión Latente 1')
    ax.set_ylabel('Dimensión Latente 2')
    ax.set_zlabel('Dimensión Latente 3')


def graficar_vecindario_3d(ax, df_3d: pd.DataFrame, palabra_objetivo: str, titulo: str) -> None:
    """
    Renderiza un gráfico 3D enfocándose en una palabra central y sus vecinos.
    """
    if df_3d.empty:
        return

    # Separar la palabra objetivo de los vecinos
    df_objetivo = df_3d[df_3d['palabra'] == palabra_objetivo]
    df_vecinos = df_3d[df_3d['palabra'] != palabra_objetivo]

    # Dibujar vecinos (Puntos azules más sutiles)
    ax.scatter(df_vecinos['x'], df_vecinos['y'], df_vecinos['z'], 
               c='steelblue', s=50, alpha=0.6)

    # Dibujar palabra objetivo (Punto rojo y destacado)
    ax.scatter(df_objetivo['x'], df_objetivo['y'], df_objetivo['z'], 
               c='crimson', s=120, edgecolors='black', linewidths=1.5)

    # Colocar etiquetas con formato dinámico
    for i, row in df_3d.iterrows():
        es_objetivo = row['palabra'] == palabra_objetivo
        peso = 'bold' if es_objetivo else 'normal'
        color = 'black' if es_objetivo else 'dimgray'
        
        ax.text(row['x'], row['y'], row['z'], f" {row['palabra']}", 
                size=11, weight=peso, color=color)

    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.set_zlabel('Dimensión 3')