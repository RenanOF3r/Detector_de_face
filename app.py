# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import traceback # Para logs de erro detalhados

# --- Configuração do MediaPipe (fora da função para eficiência) ---
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # Configura o detector:
    # model_selection=0: para rostos próximos (até 2m), geralmente melhor para fotos
    # min_detection_confidence: limiar de confiança (0.3 mostrou bons resultados)
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    print("Detector MediaPipe inicializado com sucesso.") # Log para confirmação
except AttributeError as e:
     # Erro comum se o mediapipe não estiver instalado corretamente ou for versão incompatível
     st.error(f"Erro Crítico: Falha ao inicializar componentes do MediaPipe. Verifique a instalação. Detalhes: {e}")
     st.stop() # Impede a execução do restante do app se o detector falhar
except Exception as e:
     st.error(f"Erro Crítico: Erro inesperado ao inicializar o MediaPipe. Detalhes: {e}")
     st.stop()

# --- Função de Rotação ---
def rotacionar_imagem(imagem_np, angulo):
    """Rotaciona uma imagem NumPy (BGR ou RGB)."""
    if angulo == 0: return imagem_np
    elif angulo == 90: return cv2.rotate(imagem_np, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180: return cv2.rotate(imagem_np, cv2.ROTATE_180)
    elif angulo == 270: return cv2.rotate(imagem_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return imagem_np # Retorna original para ângulos inesperados

# --- Função de Transformação de Coordenadas ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    """
    Transforma as coordenadas de uma bounding box detectada na imagem rotacionada
    de volta para o sistema de coordenadas da imagem original. Retorna inteiros.
    """
    try:
        xmin_r, ymin_r, w_r, h_r = box_rotada
        h_orig, w_orig = dim_originais
        h_rot, w_rot = dim_rotadas

        # Calcula os cantos na imagem rotacionada
        xmax_r = xmin_r + w_r
        ymax_r = ymin_r + h_r

        # Garante que os cantos não saiam dos limites da imagem rotacionada
        xmin_r = max(0, xmin_r)
        ymin_r = max(0, ymin_r)
        xmax_r = min(w_rot, xmax_r)
        ymax_r = min(h_rot, ymax_r)

        # Aplica a transformação inversa baseada no ângulo
        if angulo_rotacao == 0: xmin_o, ymin_o, xmax_o, ymax_o = xmin_r, ymin_r, xmax_r, ymax_r
        elif angulo_rotacao == 90: xmin_o, ymin_o, xmax_o, ymax_o = ymin_r, w_rot - xmax_r, ymax_r, w_rot - xmin_r
        elif angulo_rotacao == 180: xmin_o, ymin_o, xmax_o, ymax_o = w_rot - xmax_r, h_rot - ymax_r, w_rot - xmin_r, h_rot - ymin_r
        elif angulo_rotacao == 270: xmin_o, ymin_o, xmax_o, ymax_o = h_rot - ymax_r, xmin_r, h_rot - ymin_r, xmax_r
        else: return None # Ângulo inválido

        # Converte para inteiros APÓS a transformação geométrica
        xmin_o, ymin_o, xmax_o, ymax_o = int(round(xmin_o)), int(round(ymin_o)), int(round(xmax_o)), int(round(ymax_o))

        # Recalcula largura e altura e garante que sejam positivos
        w_o = max(0, xmax_o - xmin_o)
        h_o = max(0, ymax_o - ymin_o)

        # Garante que as coordenadas finais estejam dentro dos limites da imagem original
        xmin_o = max(0, xmin_o)
        ymin_o = max(0, ymin_o)
        xmin_o = min(w_orig - 1, xmin_o) if w_orig > 0 else 0
        ymin_o = min(h_orig - 1, ymin_o) if h_orig > 0 else 0

        # Recalcula w/h após ajuste de xmin/ymin para garantir que não excedam os limites
        w_o = max(0, xmax_o - xmin_o)
