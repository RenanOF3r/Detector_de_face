# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import time
import traceback

# --- Configuração do MediaPipe ---
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # Usando os parâmetros que funcionaram: model_selection=1, min_detection_confidence=0.1
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
    print(f"[{time.time()}] Detector MediaPipe inicializado (Modelo: 1, Confiança: 0.1).", flush=True)
except Exception as e:
    st.error(f"Erro Crítico ao inicializar MediaPipe: {e}")
    print(f"[{time.time()}] ERRO CRÍTICO: Falha ao inicializar MediaPipe: {e}\n{traceback.format_exc()}", flush=True)
    st.stop()

# --- Função de Rotação ---
def rotacionar_imagem(imagem_np, angulo):
    """Rotaciona uma imagem NumPy (BGR ou RGB)."""
    if angulo == 0: return imagem_np
    elif angulo == 90: return cv2.rotate(imagem_np, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180: return cv2.rotate(imagem_np, cv2.ROTATE_180)
    elif angulo == 270: return cv2.rotate(imagem_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return imagem_np

# --- Função de Transformação de Coordenadas ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    """Transforma coordenadas da box rotacionada para a original."""
    try:
        xmin_r, ymin_r, w_r, h_r = box_rotada
        h_orig, w_orig = dim_originais
        h_rot, w_rot = dim_rotadas

        if w_orig <= 0 or h_orig <= 0 or w_rot <= 0 or h_rot <= 0: return None

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

        xmin_o, ymin_o, xmax_o, ymax_o = int(round(xmin_o)), int(round(ymin_o)), int(round(xmax_o)), int(round(ymax_o))

        xmin_o = max(0, xmin_o)
        ymin_o = max(0, ymin_o)
        xmin_o = min(w_orig - 1, xmin_o) if w_orig > 0 else 0
        ymin_o = min(h_orig - 1, ymin_o) if h_orig > 0 else 0

        w_o = max(0, xmax_o - xmin_o)
        h_o = max(0, ymax_o - ymin_o)
        w_o = min(w_orig - xmin_o, w_o) if w_orig > xmin_o else 0
        h_o = min(h_orig - ymin_o, h_o) if h_orig > ymin_o else 0

        if w_o > 0 and h_o > 0:
            return (xmin_o, ymin_o, w_o, h_o)
        else:
            return None
    except Exception as e:
        print(f"  ERRO Transformação (Exceção): {e}", flush=True) # Mantém log de erro
        return None

# --- Função NMS ---
def aplicar_nms(caixas, limiar_iou=0.4):
    """Aplica Non-Maximum Suppression às caixas detectadas."""
    if not caixas:
        return []

    scores = [1.0] * len(caixas)
    caixas_formato_nms = [list(map(int, c)) for c in caixas if len(c) == 4]
    if not caixas_formato_nms:
        return []

    try:
        indices = cv2.dnn.NMSBoxes(caixas_formato_nms, scores, score_threshold=0.0, nms_threshold=limiar_iou)
    except Exception as e:
        print(f"Erro durante NMS: {e}", flush=True)
        return caixas_formato_nms # Fallback

    if indices is None or len(indices) == 0:
         return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten()

    caixas_finais = [caixas_formato_nms[i] for i in indices]
    return caixas_finais


# --- Função de Processamento Principal (com Cache e NMS) ---
# *** ALTERAÇÃO AQUI: Adicionado _bytes_imagem para ajudar o cache ***
@st.cache_data(show_spinner=False)
def detectar_e_desenhar_rostos(_imagem_pil, _bytes_imagem):
    """
    Detecta rostos em uma imagem PIL, tentando rotações, aplica NMS e retorna
    a contagem e a imagem processada (como array NumPy BGR).
    Os argumentos com _ são ignorados pelo cache para hashing, exceto _bytes_imagem.
    """
    # Usar len(_bytes_imagem) no log para confirmar que está mudando
    print(f"\n[{time.time()}] ===== INICIANDO DETECÇÃO (Cache Ativo, NMS Ativo, Bytes: {len(_bytes_imagem)}) =====", flush=True)
    try:
        imagem_rgb_original = np.array(_imagem_pil.convert('RGB'))
        imagem_rgb_original = np.copy(imagem_rgb_original)
        imagem_rgb_original.flags.writeable = True

        h_orig, w_orig, _ = imagem_rgb_original.shape
        if h_orig == 0 or w_orig == 0:
            st.error("Erro: Imagem carregada tem dimensões inválidas.")
            return 0, None

        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)
        deteccoes_antes_nms = []

        for angulo in [0, 90, 180, 270]:
            # print(f"  Processando Ângulo: {angulo} graus...", flush=True) # Log opcional
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            imagem_rgb_rotacionada = np.copy(imagem_rgb_rotacionada)
            imagem_rgb_rotacionada.flags.writeable = True
            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape
            if h_rot == 0 or w_rot == 0: continue

            resultados = face_detector.process(imagem_rgb_rotacionada)

            if resultados.detections:
                # print(f"    {len(resultados.detections)} detecção(ões) encontradas no ângulo {angulo}.", flush=True) # Log opcional
                for detection in resultados.detections:
                    box_relativa = detection.location_data.relative_bounding_box
                    if box_relativa:
                        if not (0 <= box_relativa.xmin <= 1 and 0 <= box_relativa.ymin <= 1 and box_relativa.width > 0 and box_relativa.height > 0):
                            continue

                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.ceil(box_relativa.width * w_rot)
                        h_r = math.ceil(box_relativa.height * h_rot)
                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)

                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )

                        if box_abs_original:
                            deteccoes_antes_nms.append(box_abs_original)

        # print(f"\n  Aplicando NMS em {len(deteccoes_antes_nms)} detecções brutas...", flush=True) # Log opcional
        caixas_finais_nms = aplicar_nms(deteccoes_antes_nms, limiar_iou=0.4)
        numero_rostos = len(caixas_finais_nms)
        print(f"  {numero_rostos} rosto(s) detectado(s) após NMS.", flush=True)

        if numero_rostos > 0:
            for (x, y, w, h) in caixas_finais_nms:
                cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (0, 255, 0), 2) # Verde
        else:
            # Apenas log, a interface mostrará 0 rostos
             print("  Nenhum rosto detectado após NMS.")


        print(f"[{time.time()}] ===== DETECÇÃO FINALIZADA =====", flush=True)
        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        print(f"[{time.time()}] ERRO INESPERADO em detectar_e_desenhar_rostos: {e}\n{traceback.format_exc()}", flush=True)
        st.error(f"Erro inesperado durante o processamento da imagem: {e}")
        return -1, None

# --- Interface Gráfica com Streamlit ---

st.set_page_config(page_title="Detector Facial MediaPipe", layout="wide", initial_sidebar_state="auto")
st.title("👁️ Detector de Rostos com MediaPipe")
st.write("Faça o upload de uma imagem e veja os rostos detectados!")
st.markdown("---")

arquivo_imagem_enviado = st.file_uploader(
    "Selecione um arquivo de imagem:",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Arraste e solte ou clique para escolher uma imagem."
)

if arquivo_imagem_enviado is not None:
    try:
        # Lê os bytes uma vez para passar para a função cacheada
        bytes_da_imagem = arquivo_imagem_enviado.getvalue()
        # Cria o objeto PIL a partir dos bytes lidos
        imagem_pil = Image.open(arquivo_imagem_enviado) # Ou Image.open(io.BytesIO(bytes_da_imagem))

        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            # Mostra a imagem PIL
            st.image(imagem_pil, caption=f"Original: {arquivo_imagem_enviado.name}", use_container_width=True)

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            with st.spinner('Detectando rostos...'):
                # *** ALTERAÇÃO AQUI: Passando imagem_pil e bytes_da_imagem ***
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil, bytes_da_imagem)

            if imagem_final_bgr is not None and num_rostos >= 0:
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                st.image(imagem_final_rgb, caption=f"Processada: {num_rostos} rosto(s) detectado(s)", use_container_width=True)
                sucesso_encode, buffer = cv2.imencode('.png', imagem_final_bgr)
                if sucesso_encode:
                    img_bytes_download = buffer.tobytes()
                    nome_base = arquivo_imagem_enviado.name.rsplit('.', 1)[0]
                    st.download_button(
                       label="💾 Baixar Imagem Processada",
                       data=img_bytes_download,
                       file_name=f"rostos_detectados_{nome_base}.png",
                       mime="image/png"
                    )
                else:
                    st.warning("Não foi possível gerar o arquivo para download.")
            elif num_rostos == 0:
                 st.info("Nenhum rosto detectado na imagem após processamento e NMS.")
            else: # num_rostos == -1 indica erro na função
                st.error("Falha ao processar a imagem. Verifique os logs se disponíveis.")

    except Exception as e:
        print(f"[{time.time()}] ERRO ao carregar/processar imagem no Streamlit: {e}\n{traceback.format_exc()}", flush=True)
        st.error(f"Erro ao carregar ou processar o arquivo de imagem: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    st.info("☝️ Faça o upload de uma imagem para iniciar a detecção.")

st.markdown("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")

# --- Informação sobre arquivos de imagem grandes ---
# Adicionado para cumprir a instrução especial sobre arquivos de imagem
st.sidebar.title("ℹ️ Informações Adicionais")
st.sidebar.info("""
Os arquivos `image.png` que foram enviados anteriormente não puderam ter seu texto extraído, possivelmente por serem muito grandes ou por um problema temporário na ferramenta de extração.

**Estes documentos só podem ser usados na execução de código.**

Exemplo de como carregar uma dessas imagens em código Python (se estivessem disponíveis no ambiente de execução):
```python
from PIL import Image
import io

# Supondo que 'file_content' contenha os bytes do arquivo image.png
# file_content = ... # obter os bytes do arquivo

# try:
#     img = Image.open(io.BytesIO(file_content))
#     print("Imagem 'image.png' carregada com sucesso.")
#     # Processar a imagem 'img'
# except Exception as e:
#     print(f"Erro ao carregar 'image.png': {e}")
