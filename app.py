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
    # print("Detector MediaPipe inicializado com sucesso.") # Log para confirmação (opcional)
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
        w_o = max(0, xmax_o - xmin_o)
        h_o = max(0, ymax_o - ymin_o)
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
        st.warning(f"Aviso: Erro durante a transformação de coordenadas: {e}")
        return None


# --- Função de Processamento Principal (com argumento ignorado no cache) ---
# Usar cache do Streamlit para evitar reprocessar a mesma imagem se nada mudar
@st.cache_data(show_spinner=False) # show_spinner=False pois temos nosso próprio spinner
# CORREÇÃO APLICADA AQUI: _imagem_pil
def detectar_e_desenhar_rostos(_imagem_pil):
    """
    Detecta rostos em uma imagem PIL, tentando rotações, aplica NMS e retorna
    a contagem e a imagem processada (como array NumPy BGR).
    O argumento _imagem_pil é ignorado pelo cache do Streamlit.
    """
    try:
        # CORREÇÃO APLICADA AQUI: usa _imagem_pil
        imagem_rgb_original = np.array(_imagem_pil.convert('RGB'))
        imagem_rgb_original = np.copy(imagem_rgb_original)
        imagem_rgb_original.flags.writeable = True

        h_orig, w_orig, _ = imagem_rgb_original.shape
        if h_orig == 0 or w_orig == 0:
             st.error("Erro: Imagem carregada tem dimensões inválidas.")
             return 0, None

        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)
        deteccoes_finais_orig = []

        for angulo in [0, 90, 180, 270]:
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            imagem_rgb_rotacionada = np.copy(imagem_rgb_rotacionada)
            imagem_rgb_rotacionada.flags.writeable = True
            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape
            if h_rot == 0 or w_rot == 0:
                st.warning(f"Aviso: Imagem rotacionada para {angulo} graus tem dimensões inválidas.")
                continue

            resultados = face_detector.process(imagem_rgb_rotacionada)

            if resultados.detections:
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
                            deteccoes_finais_orig.append(box_abs_original)

        numero_rostos = 0
        if deteccoes_finais_orig:
            deteccoes_validas = [d for d in deteccoes_finais_orig if isinstance(d, tuple) and len(d) == 4 and all(isinstance(v, int) and v >= 0 for v in d) and d[2] > 0 and d[3] > 0]
            if not deteccoes_validas:
                 st.info("Nenhuma detecção válida encontrada após transformações.")
            else:
                boxes_para_nms = np.array([[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in deteccoes_validas], dtype=np.float32)
                confiancas_ficticias = np.ones(len(boxes_para_nms), dtype=np.float32)
                nms_threshold = 0.3
                score_threshold = 0.1
                indices_mantidos = cv2.dnn.NMSBoxes(boxes_para_nms.tolist(), confiancas_ficticias.tolist(), score_threshold=score_threshold, nms_threshold=nms_threshold)

                if len(indices_mantidos) > 0:
                     if isinstance(indices_mantidos, tuple):
                         indices_finais = indices_mantidos[0] if len(indices_mantidos) > 0 else []
                     else:
                         indices_finais = indices_mantidos.flatten()
                     numero_rostos = len(indices_finais)
                     st.success(f"**Número final de rostos detectados: {numero_rostos}**")
                     for i in indices_finais:
                         x, y, w, h = deteccoes_validas[i]
                         cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (36, 255, 12), 3)
                else:
                     st.info("**Nenhum rosto detectado após filtro de sobreposição (NMS).**")
                     numero_rostos = 0
        else:
            st.info("**Nenhum rosto detectado na imagem (em nenhuma rotação).**")

        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        st.error(f"Erro inesperado durante o processamento da imagem: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return -1, None

# --- Interface Gráfica com Streamlit ---

st.set_page_config(page_title="Detector Facial MediaPipe", layout="wide", initial_sidebar_state="auto")
st.title("👁️ Detector de Rostos com MediaPipe")
st.write("Faça o upload de uma imagem (mesmo com rostos rotacionados) e veja quantos são detectados!")
st.markdown("---")

arquivo_imagem_enviado = st.file_uploader(
    "Selecione um arquivo de imagem:",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Arraste e solte ou clique para escolher uma imagem."
)

if arquivo_imagem_enviado is not None:
    try:
        imagem_pil = Image.open(arquivo_imagem_enviado)
        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            st.image(imagem_pil, caption=f"Original: {arquivo_imagem_enviado.name}", use_container_width=True)

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            with st.spinner('Detectando rostos (testando rotações)... Aguarde!'):
                # CORREÇÃO APLICADA AQUI: passa imagem_pil para a função
                # O cache funcionará corretamente agora devido ao '_' no nome do argumento
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)

            if imagem_final_bgr is not None and num_rostos >= 0:
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                st.image(imagem_final_rgb, caption=f"Processada: {num_rostos} rosto(s) detectado(s)", use_container_width=True)
                sucesso_encode, buffer = cv2.imencode('.png', imagem_final_bgr)
                if sucesso_encode:
                    img_bytes = buffer.tobytes()
                    nome_base = arquivo_imagem_enviado.name.rsplit('.', 1)[0]
                    st.download_button(
                       label="💾 Baixar Imagem Processada",
                       data=img_bytes,
                       file_name=f"rostos_detectados_{nome_base}.png",
                       mime="image/png"
                    )
                else:
                    st.warning("Não foi possível gerar o arquivo para download.")
            elif num_rostos == 0:
                 st.info("Processamento concluído, mas nenhum rosto foi detectado nesta imagem.")
            else:
                st.error("Falha ao processar a imagem. Verifique os logs se disponíveis.")

    except Exception as e:
        st.error(f"Erro ao carregar ou processar o arquivo de imagem: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    st.info("☝️ Faça o upload de uma imagem para iniciar a detecção.")

st.markdown("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")
