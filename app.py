# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math # Para cálculos de coordenadas

# --- Configuração do MediaPipe (fora da função para eficiência) ---
# Inicializa a solução de detecção de rosto
mp_face_detection = mp.solutions.face_detection
# Inicializa utilitários para desenho
mp_drawing = mp.solutions.drawing_utils
# Configura o detector:
# model_selection=0: para rostos próximos (até 2m), pode ser melhor para fotos
# model_selection=1: para rostos mais distantes (até 5m)
# min_detection_confidence: limiar de confiança (0.5 é um bom começo)
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
        return imagem_np # Retorna original para ângulos inesperados

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
        tuple: (xmin, ymin, width, height) absolutos na imagem original, ou None se inválido.
    """
    xmin_r, ymin_r, w_r, h_r = box_rotada
    h_orig, w_orig = dim_originais
    h_rot, w_rot = dim_rotadas

    # Calcula os cantos na imagem rotacionada
    xmax_r = xmin_r + w_r
    ymax_r = ymin_r + h_r

    # Garante que as coordenadas não saiam dos limites da imagem rotacionada
    xmin_r = max(0, xmin_r)
    ymin_r = max(0, ymin_r)
    xmax_r = min(w_rot, xmax_r)
    ymax_r = min(h_rot, ymax_r)

    if angulo_rotacao == 0:
        xmin_o, ymin_o, xmax_o, ymax_o = xmin_r, ymin_r, xmax_r, ymax_r

    elif angulo_rotacao == 90: # Rotacionado 90 graus horário
        xmin_o = ymin_r
        ymin_o = w_rot - xmax_r # w_rot é a altura original h_orig
        xmax_o = ymax_r
        ymax_o = w_rot - xmin_r

    elif angulo_rotacao == 180: # Rotacionado 180 graus
        xmin_o = w_rot - xmax_r # w_rot é w_orig
        ymin_o = h_rot - ymax_r # h_rot é h_orig
        xmax_o = w_rot - xmin_r
        ymax_o = h_rot - ymin_r

    elif angulo_rotacao == 270: # Rotacionado 270 graus horário (90 anti-horário)
        xmin_o = h_rot - ymax_r # h_rot é a largura original w_orig
        ymin_o = xmin_r
        xmax_o = h_rot - ymin_r
        ymax_o = xmax_r
    else:
        return None # Ângulo inválido

    # Recalcula largura e altura e garante que sejam positivos
    w_o = max(0, xmax_o - xmin_o)
    h_o = max(0, ymax_o - ymin_o)

    # Garante que as coordenadas finais estejam dentro dos limites da imagem original
    xmin_o = max(0, xmin_o)
    ymin_o = max(0, ymin_o)
    xmin_o = min(w_orig - 1, xmin_o) # -1 pois coordenadas são base 0
    ymin_o = min(h_orig - 1, ymin_o)
    w_o = min(w_orig - xmin_o, w_o)
    h_o = min(h_orig - ymin_o, h_o)

    # Retorna apenas se a caixa tiver tamanho válido
    if w_o > 0 and h_o > 0:
        return (xmin_o, ymin_o, w_o, h_o)
    else:
        return None


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

        deteccoes_finais_orig = [] # Lista para guardar boxes (x, y, w, h) nas coordenadas originais

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
                        # Usar math.floor/ceil pode ser mais robusto que int() direto
                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.ceil(box_relativa.width * w_rot)
                        h_r = math.ceil(box_relativa.height * h_rot)

                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)

                        # Transformar as coordenadas de volta para a imagem ORIGINAL
                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )

                        if box_abs_original:
                            deteccoes_finais_orig.append(box_abs_original)

        # --- Pós-processamento: Non-Maximum Suppression (NMS) ---
        numero_rostos = 0
        if deteccoes_finais_orig:
            # Converter boxes (x, y, w, h) para (x1, y1, x2, y2) para NMS
            boxes_para_nms = np.array([[x, y, x + w, y + h] for x, y, w, h in deteccoes_finais_orig], dtype=np.float32) # Precisa ser float
            confiancas_ficticias = np.ones(len(boxes_para_nms), dtype=np.float32) # Precisa ser float

            # Limiar de sobreposição (IoU)
            nms_threshold = 0.3
            # NMS retorna os *índices* das boxes a serem mantidas
            indices_mantidos = cv2.dnn.NMSBoxes(boxes_para_nms.tolist(), confiancas_ficticias.tolist(), score_threshold=0.1, nms_threshold=nms_threshold)

            if len(indices_mantidos) > 0:
                 # Achatamos o array de índices caso ele venha como [[0], [2], ...]
                 indices_finais = indices_mantidos.flatten()
                 numero_rostos = len(indices_finais)
                 st.success(f"**Número de rostos detectados (após NMS): {numero_rostos}**")

                 # Desenhar apenas as boxes mantidas pelo NMS
                 for i in indices_finais:
                     # Pegar a box original (x, y, w, h) correspondente ao índice mantido
                     x, y, w, h = deteccoes_finais_orig[i]
                     # Desenhar retângulo na imagem BGR de desenho
                     cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (0, 255, 0), 3) # Verde, mais espesso
            else:
                 # Se NMS não retornou nada, mas havia detecções
                 st.info("**Nenhum rosto detectado após filtro NMS.**")
                 numero_rostos = 0
        else:
            st.info("**Nenhum rosto detectado na imagem (em nenhuma rotação).**")

        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        st.error(f"Erro durante o processamento da imagem com MediaPipe (com rotações): {e}")
        import traceback
        st.error(traceback.format_exc()) # Mostra mais detalhes do erro para debug
        return None, None

# --- Interface Gráfica com Streamlit ---

# Configurações da página
st.set_page_config(page_title="Detector Facial MediaPipe (com Rotação)", layout="wide", initial_sidebar_state="collapsed")

# Título da Aplicação
st.title("👁️ Detector de Rostos com MediaPipe (Robusto à Rotação)")
st.write("Faça o upload de uma imagem (mesmo com rostos rotacionados) e veja quantos são detectados!")

# Componente para upload de arquivo
arquivo_imagem_enviado = st.file_uploader(
    "Selecione um arquivo de imagem:",
    type=["jpg", "jpeg", "png", "bmp", "webp"], # Formatos de imagem permitidos
    help="Arraste e solte ou clique para escolher uma imagem."
)

# Verifica se um arquivo foi enviado
if arquivo_imagem_enviado is not None:
    try:
        # Abrir a imagem usando a biblioteca PIL
        imagem_pil = Image.open(arquivo_imagem_enviado)

        st.write("---") # Linha separadora

        # Criar duas colunas para exibir as imagens lado a lado
        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            # Exibir a imagem original - CORRIGIDO
            st.image(imagem_pil, caption="Imagem enviada", use_container_width=True)

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            # Mostrar um spinner enquanto processa
            with st.spinner('Detectando rostos (testando rotações)... Aguarde!'):
                # Chamar a função de detecção
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)

            # Se o processamento foi bem-sucedido
            if imagem_final_bgr is not None:
                # Converter a imagem processada de BGR (OpenCV) para RGB (Streamlit/PIL)
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                # Exibir a imagem processada - CORRIGIDO
                st.image(imagem_final_rgb, caption=f"Detecção concluída ({num_rostos} rosto(s))", use_container_width=True)

                # Oferecer botão para download da imagem processada
                # Converter array NumPy BGR para bytes para download
                # Usar PNG para evitar perda de qualidade
                sucesso_encode, buffer = cv2.imencode('.png', imagem_final_bgr)
                if sucesso_encode:
                    img_bytes = buffer.tobytes()
                    st.download_button(
                       label="Baixar Imagem Processada",
                       data=img_bytes,
                       # Usar o nome original do arquivo na sugestão de nome para download
                       file_name=f"rostos_detectados_{arquivo_imagem_enviado.name.split('.')[0]}.png",
                       mime="image/png"
                    )
                else:
                    st.warning("Não foi possível gerar o arquivo para download.")
            else:
                # Mensagem se o processamento falhou dentro da função
                st.error("Não foi possível processar a imagem.")

    except Exception as e:
        # Captura erros ao tentar abrir a imagem (ex: arquivo corrompido)
        st.error(f"Erro ao carregar ou processar o arquivo: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    # Mensagem inicial quando nenhum arquivo foi enviado ainda
    st.info("Aguardando o upload de uma imagem para iniciar a detecção.")

# Rodapé
st.write("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")
