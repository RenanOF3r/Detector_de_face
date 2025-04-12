import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import os

# --- Configuração do MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Usando confiança reduzida e modelo para rostos próximos/médios
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

# --- Função de Rotação ---
def rotacionar_imagem(imagem_np, angulo):
    if angulo == 0:
        return imagem_np
    elif angulo == 90:
        return cv2.rotate(imagem_np, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180:
        return cv2.rotate(imagem_np, cv2.ROTATE_180)
    elif angulo == 270:
        return cv2.rotate(imagem_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return imagem_np

# --- Função de Transformação de Coordenadas ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    xmin_r, ymin_r, w_r, h_r = box_rotada
    h_orig, w_orig = dim_originais
    h_rot, w_rot = dim_rotadas
    xmax_r = xmin_r + w_r
    ymax_r = ymin_r + h_r
    xmin_r = max(0, xmin_r)
    ymin_r = max(0, ymin_r)
    xmax_r = min(w_rot, xmax_r)
    ymax_r = min(h_rot, ymax_r)

    if angulo_rotacao == 0: xmin_o, ymin_o, xmax_o, ymax_o = xmin_r, ymin_r, xmax_r, ymax_r
    elif angulo_rotacao == 90: xmin_o, ymin_o, xmax_o, ymax_o = ymin_r, w_rot - xmax_r, ymax_r, w_rot - xmin_r
    elif angulo_rotacao == 180: xmin_o, ymin_o, xmax_o, ymax_o = w_rot - xmax_r, h_rot - ymax_r, w_rot - xmin_r, h_rot - ymin_r
    elif angulo_rotacao == 270: xmin_o, ymin_o, xmax_o, ymax_o = h_rot - ymax_r, xmin_r, h_rot - ymin_r, xmax_r
    else: return None

    w_o = max(0, xmax_o - xmin_o)
    h_o = max(0, ymax_o - ymin_o)
    xmin_o = max(0, xmin_o)
    ymin_o = max(0, ymin_o)
    xmin_o = min(w_orig - 1, xmin_o)
    ymin_o = min(h_orig - 1, ymin_o)
    w_o = min(w_orig - xmin_o, w_o)
    h_o = min(h_orig - ymin_o, h_o)

    if w_o > 0 and h_o > 0: return (xmin_o, ymin_o, w_o, h_o)
    else: return None

# --- Lógica Principal de Detecção (Adaptada) ---
def detectar_em_arquivo(caminho_imagem_entrada, caminho_imagem_saida):
    try:
        print(f"Carregando imagem: {caminho_imagem_entrada}")
        if not os.path.exists(caminho_imagem_entrada):
             print(f"Erro: Arquivo de imagem não encontrado em '{caminho_imagem_entrada}'")
             return 0

        imagem_pil = Image.open(caminho_imagem_entrada)
        imagem_rgb_original = np.array(imagem_pil.convert('RGB'))
        h_orig, w_orig, _ = imagem_rgb_original.shape
        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)
        deteccoes_finais_orig = []

        print("Iniciando detecção com rotações...")
        for angulo in [0, 90, 180, 270]:
            print(f"  Testando ângulo: {angulo} graus")
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape
            resultados = face_detector.process(imagem_rgb_rotacionada)

            if resultados.detections:
                print(f"    Detecções encontradas no ângulo {angulo}: {len(resultados.detections)}")
                for detection in resultados.detections:
                    box_relativa = detection.location_data.relative_bounding_box
                    if box_relativa:
                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.ceil(box_relativa.width * w_rot)
                        h_r = math.ceil(box_relativa.height * h_rot)
                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)
                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )
                        if box_abs_original:
                            deteccoes_finais_orig.append(box_abs_original)
            else:
                 print(f"    Nenhuma detecção no ângulo {angulo}.")


        print("Aplicando NMS...")
        numero_rostos = 0
        if deteccoes_finais_orig:
            boxes_para_nms = np.array([[x, y, x + w, y + h] for x, y, w, h in deteccoes_finais_orig], dtype=np.float32)
            confiancas_ficticias = np.ones(len(boxes_para_nms), dtype=np.float32)
            nms_threshold = 0.3
            indices_mantidos = cv2.dnn.NMSBoxes(boxes_para_nms.tolist(), confiancas_ficticias.tolist(), score_threshold=0.1, nms_threshold=nms_threshold)

            if len(indices_mantidos) > 0:
                 indices_finais = indices_mantidos.flatten()
                 numero_rostos = len(indices_finais)
                 print(f"Número final de rostos detectados (após NMS): {numero_rostos}")
                 for i in indices_finais:
                     x, y, w, h = deteccoes_finais_orig[i]
                     cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                 print("Nenhum rosto detectado após filtro NMS.")
                 numero_rostos = 0
        else:
            print("Nenhum rosto detectado em nenhuma rotação.")

        print(f"Salvando resultado em: {caminho_imagem_saida}")
        sucesso_escrita = cv2.imwrite(caminho_imagem_saida, imagem_bgr_para_desenho)
        if not sucesso_escrita:
            print("Erro ao salvar a imagem resultante.")
        else:
            print("Imagem resultante salva com sucesso.")

        return numero_rostos

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        import traceback
        print(traceback.format_exc())
        return -1 # Indica erro

# --- Execução ---
caminho_entrada = 'sua_imagem.jpg'
caminho_saida = 'resultado_deteccao.png'
contagem_final = detectar_em_arquivo(caminho_entrada, caminho_saida)

if contagem_final >= 0:
    print(f"\nProcesso concluído. Total de rostos detectados: {contagem_final}")
else:
    print("\nProcesso concluído com erro.")
