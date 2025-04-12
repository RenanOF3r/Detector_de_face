# app.py (modificado)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math # Para cálculos de coordenadas

# --- Configuração do MediaPipe (fora da função para eficiência) ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Usar model_selection=0 pode ser melhor para fotos onde o rosto ocupa mais espaço
# Ajuste min_detection_confidence se necessário (0.5 é um bom começo)
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Função de Rotação ---
def rotacionar_imagem(imagem_np, angulo):
    """Rotaciona uma imagem NumPy (BGR ou RGB)."""
    if angulo == 0:
        return imagem_np
    elif angulo == 90:
        return cv2.rotate(imagem_np, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180:
        return cv2.rotate(imagem_np, cv2.ROTATE_180)
    elif angulo == 270:
        return cv2.rotate(imagem_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Para outros ângulos, seria mais complexo (mas não necessário aqui)
        return imagem_np

# --- Função de Transformação de Coordenadas ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    """
    Transforma as coordenadas de uma bounding box detectada na imagem rotacionada
    de volta para o sistema de coordenadas da imagem original.

    Args:
        box_rotada (tuple): (xmin, ymin, width, height) absolutos na imagem rotacionada.
        angulo_rotacao (int): 0, 90, 180, 270.
        dim_originais (tuple): (altura_orig, largura_orig).
        dim_rotadas (tuple): (altura_rot, largura_rot).

    Returns:
        tuple: (xmin, ymin, width, height) absolutos na imagem original.
    """
    xmin_r, ymin_r, w_r, h_r = box_rotada
    h_orig, w_orig = dim_originais
    h_rot, w_rot = dim_rotadas # Apenas para clareza, poderiam ser recalculadas

    # Calcula os cantos na imagem rotacionada
    xmax_r = xmin_r + w_r
    ymax_r = ymin_r + h_r

    if angulo_rotacao == 0:
        return box_rotada # Nenhuma transformação necessária

    elif angulo_rotacao == 90: # Rotacionado 90 graus horário
        # (x_orig, y_orig) = (y_rot, w_rot - 1 - x_rot) para um ponto
        xmin_o = ymin_r
        ymin_o = w_rot - xmax_r # w_rot é a altura original h_orig
        xmax_o = ymax_r
        ymax_o = w_rot - xmin_r
        w_o = xmax_o - xmin_o
        h_o = ymax_o - ymin_o
        return (xmin_o, ymin_o, w_o, h_o)

    elif angulo_rotacao == 180: # Rotacionado 180 graus
        # (x_orig, y_orig) = (w_rot - 1 - x_rot, h_rot - 1 - y_rot)
        xmin_o = w_rot - xmax_r # w_rot é w_orig
        ymin_o = h_rot - ymax_r # h_rot é h_orig
        xmax_o = w_rot - xmin_r
        ymax_o = h_rot - ymin_r
        w_o = xmax_o - xmin_o
        h_o = ymax_o - ymin_o
        return (xmin_o, ymin_o, w_o, h_o)

    elif angulo_rotacao == 270: # Rotacionado 270 graus horário (90 anti-horário)
        # (x_orig, y_orig) = (h_rot - 1 - y_rot, x_rot)
        xmin_o = h_rot - ymax_r # h_rot é a largura original w_orig
        ymin_o = xmin_r
        xmax_o = h_rot - ymin_r
        ymax_o = xmax_r
        w_o = xmax_o - xmin_o
        h_o = ymax_o - ymin_o
        return (xmin_o, ymin_o, w_o, h_o)

    return None # Caso de ângulo inesperado

# --- Função de Processamento Modificada ---
def detectar_e_desenhar_rostos(imagem_pil):
    """
    Detecta rostos em uma imagem PIL, tentando rotações, e retorna a contagem
    e a imagem processada (como array NumPy BGR).
    """
    try:
        # Converter a imagem PIL para um array NumPy no formato RGB
        imagem_rgb_original = np.array(imagem_pil.convert('RGB'))
        h_orig, w_orig, _ = imagem_rgb_original.shape

        # Criar uma cópia BGR para desenho final
        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)

        deteccoes_finais_orig = [] # Lista para guardar boxes nas coordenadas originais

        # Iterar sobre as rotações
        for angulo in [0, 90, 180, 270]:
            # Rotacionar a imagem RGB
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape

            # Processar a imagem rotacionada com o MediaPipe
            resultados = face_detector.process(imagem_rgb_rotacionada)

            if resultados.detections:
                # Processar cada detecção encontrada nesta rotação
                for detection in resultados.detections:
                    # Obter a bounding box relativa
                    box_relativa = detection.location_data.relative_bounding_box
                    if box_relativa:
                        # Converter para coordenadas de pixel ABSOLUTAS na imagem ROTACIONADA
                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.floor(box_relativa.width * w_rot)
                        h_r = math.floor(box_relativa.height * h_rot)

                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)

                        # Transformar as coordenadas de volta para a imagem ORIGINAL
                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )

                        if box_abs_original:
                            deteccoes_finais_orig.append(box_abs_original)

        # --- Pós-processamento (Opcional, mas recomendado: Non-Maximum Suppression) ---
        # Se um rosto for detectado em múltiplas rotações, podemos ter boxes sobrepostas.
        # NMS ajuda a manter apenas a melhor box para cada rosto.
        # Usaremos uma implementação simples de NMS do OpenCV se disponível,
        # ou apenas desenharemos todas as boxes encontradas.

        boxes_para_nms = np.array([[x, y, x + w, y + h] for x, y, w, h in deteccoes_finais_orig], dtype=np.int32)
        confiancas_ficticias = np.ones(len(boxes_para_nms)) # NMS precisa de 'confianças'

        # Limiar de sobreposição (IoU - Intersection over Union)
        # Valores menores são mais rigorosos (menos sobreposição permitida)
        nms_threshold = 0.3
        indices_mantidos = cv2.dnn.NMSBoxes(boxes_para_nms.tolist(), confiancas_ficticias.tolist(), score_threshold=0.1, nms_threshold=nms_threshold)

        numero_rostos = 0
        if len(indices_mantidos) > 0:
             # Se NMS retornou algo, use esses índices
             # Em algumas versões/casos, indices_mantidos pode ser uma tupla ou array 2D, achatamos
             if isinstance(indices_mantidos, tuple):
                 indices_finais = indices_mantidos[0] if len(indices_mantidos) > 0 else []
             else:
                 indices_finais = indices_mantidos.flatten()

             numero_rostos = len(indices_finais)
             st.success(f"**Número de rostos detectados (após NMS): {numero_rostos}**")

             # Desenhar apenas as boxes mantidas pelo NMS
             for i in indices_finais:
                 x, y, w, h = deteccoes_finais_orig[i]
                 # Desenhar retângulo na imagem BGR de desenho
                 cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (0, 255, 0), 2) # Verde
                 # Opcional: Desenhar pontos chave (requer mais transformações)
                 # mp_drawing.draw_detection não funciona diretamente aqui pois espera a 'detection' original
        else:
             # Se NMS não retornou nada (ou não foi aplicado), verificar se havia detecções antes
             if len(deteccoes_finais_orig) > 0:
                 # Caso NMS tenha falhado ou removido tudo, desenha as originais como fallback
                 # (Isso não deveria acontecer com confianças fictícias = 1, mas por segurança)
                 numero_rostos = len(deteccoes_finais_orig)
                 st.warning(f"**NMS não retornou índices, desenhando todas as {numero_rostos} detecções encontradas.**")
                 for x, y, w, h in deteccoes_finais_orig:
                     cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (255, 0, 0), 2) # Azul (para indicar fallback)
             else:
                 st.info("**Nenhum rosto detectado na imagem (em nenhuma rotação).**")

        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        st.error(f"Erro durante o processamento da imagem com MediaPipe (com rotações): {e}")
        import traceback
        st.error(traceback.format_exc()) # Mostra mais detalhes do erro para debug
        return None, None

# --- Interface Gráfica com Streamlit (O restante do código permanece o mesmo) ---

# Configurações da página
st.set_page_config(page_title="Detector Facial MediaPipe (com Rotação)", layout="wide", initial_sidebar_state="collapsed")

# Título da Aplicação
st.title("👁️ Detector de Rostos com MediaPipe (Robusto à Rotação)")
st.write("Faça o upload de uma imagem (mesmo com rostos rotacionados) e veja quantos são detectados!")

# Componente para upload de arquivo
arquivo_imagem_enviado = st.file_uploader(
    "Selecione um arquivo de imagem:",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Arraste e solte ou clique para escolher uma imagem."
)

# Verifica se um arquivo foi enviado
if arquivo_imagem_enviado is not None:
    try:
        imagem_pil = Image.open(arquivo_imagem_enviado)
        st.write("---")
        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            st.image(imagem_pil, caption="Imagem enviada", use_column_width='always')

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            with st.spinner('Detectando rostos (testando rotações)... Aguarde!'):
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)

            if imagem_final_bgr is not None:
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                st.image(imagem_final_rgb, caption=f"Detecção concluída ({num_rostos} rosto(s))", use_column_width='always')

                img_bytes = cv2.imencode('.png', imagem_final_bgr)[1].tobytes()
                st.download_button(
                   label="Baixar Imagem Processada",
                   data=img_bytes,
                   file_name=f"rostos_detectados_{arquivo_imagem_enviado.name}.png",
                   mime="image/png"
                )
            else:
                st.error("Não foi possível processar a imagem.")

    except Exception as e:
        st.error(f"Erro ao carregar ou processar o arquivo: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    st.info("Aguardando o upload de uma imagem para iniciar a detecção.")

st.write("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")
