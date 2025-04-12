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

# --- Função de Transformação de Coordenadas (Estrutura Corrigida) ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    """
    Transforma as coordenadas de uma bounding box detectada na imagem rotacionada
    de volta para o sistema de coordenadas da imagem original. Retorna inteiros.
    """
    # Todo o bloco de lógica está dentro do try
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
        else:
            # Se o ângulo for inválido, retorna None imediatamente dentro do try
            return None

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
        # (Esta parte é importante para evitar caixas que saem da imagem)
        w_o = max(0, xmax_o - xmin_o) # Recalcula w_o com base nos x ajustados
        h_o = max(0, ymax_o - ymin_o) # Recalcula h_o com base nos y ajustados
        w_o = min(w_orig - xmin_o, w_o) if w_orig > xmin_o else 0
        h_o = min(h_orig - ymin_o, h_o) if h_orig > ymin_o else 0

        # Retorna apenas se a caixa tiver tamanho válido
        if w_o > 0 and h_o > 0:
            return (xmin_o, ymin_o, w_o, h_o)
        else:
            # Se a caixa final for inválida, retorna None
            return None

    # O bloco except está corretamente alinhado com o try
    except Exception as e:
        # Em caso de qualquer erro inesperado durante a transformação, loga e retorna None
        st.warning(f"Aviso: Erro durante a transformação de coordenadas: {e}")
        # print(traceback.format_exc()) # Descomente para debug mais detalhado nos logs
        return None


# --- Função de Processamento Principal ---
# Usar cache do Streamlit para evitar reprocessar a mesma imagem se nada mudar
@st.cache_data(show_spinner=False) # show_spinner=False pois temos nosso próprio spinner
def detectar_e_desenhar_rostos(imagem_pil):
    """
    Detecta rostos em uma imagem PIL, tentando rotações, aplica NMS e retorna
    a contagem e a imagem processada (como array NumPy BGR).
    """
    try:
        # Converter a imagem PIL para um array NumPy no formato RGB
        imagem_rgb_original = np.array(imagem_pil.convert('RGB'))

        # Garante que a imagem não seja somente leitura (necessário para MediaPipe)
        imagem_rgb_original = np.copy(imagem_rgb_original)
        imagem_rgb_original.flags.writeable = True

        h_orig, w_orig, _ = imagem_rgb_original.shape
        if h_orig == 0 or w_orig == 0:
             st.error("Erro: Imagem carregada tem dimensões inválidas.")
             return 0, None

        # Criar uma cópia BGR para desenho final
        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)

        deteccoes_finais_orig = [] # Lista para guardar boxes (x, y, w, h) nas coordenadas originais

        # Iterar sobre as rotações
        for angulo in [0, 90, 180, 270]:
            # Rotacionar a imagem RGB
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            # Garante que a imagem rotacionada seja gravável e contígua
            imagem_rgb_rotacionada = np.copy(imagem_rgb_rotacionada)
            imagem_rgb_rotacionada.flags.writeable = True

            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape
            if h_rot == 0 or w_rot == 0:
                st.warning(f"Aviso: Imagem rotacionada para {angulo} graus tem dimensões inválidas.")
                continue # Pula para o próximo ângulo

            # Processar a imagem rotacionada com o MediaPipe
            resultados = face_detector.process(imagem_rgb_rotacionada)

            if resultados.detections:
                # Processar cada detecção encontrada nesta rotação
                for detection in resultados.detections:
                    box_relativa = detection.location_data.relative_bounding_box
                    if box_relativa:
                         # Verifica validade básica da box relativa
                        if not (0 <= box_relativa.xmin <= 1 and 0 <= box_relativa.ymin <= 1 and box_relativa.width > 0 and box_relativa.height > 0):
                            continue # Ignora box relativa inválida

                        # Converter para coordenadas de pixel ABSOLUTAS na imagem ROTACIONADA
                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.ceil(box_relativa.width * w_rot)
                        h_r = math.ceil(box_relativa.height * h_rot)
                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)

                        # Transformar as coordenadas de volta para a imagem ORIGINAL
                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )

                        if box_abs_original: # Verifica se a transformação retornou uma box válida
                            deteccoes_finais_orig.append(box_abs_original)

        # --- Pós-processamento: Non-Maximum Suppression (NMS) ---
        numero_rostos = 0
        if deteccoes_finais_orig:
            # Filtra novamente por segurança, garantindo tuplas de 4 inteiros positivos
            deteccoes_validas = [d for d in deteccoes_finais_orig if isinstance(d, tuple) and len(d) == 4 and all(isinstance(v, int) and v >= 0 for v in d) and d[2] > 0 and d[3] > 0]

            if not deteccoes_validas:
                 st.info("Nenhuma detecção válida encontrada após transformações.")
            else:
                # Converte para float32 para NMS (x1, y1, x2, y2)
                boxes_para_nms = np.array([[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in deteccoes_validas], dtype=np.float32)
                # Cria confianças fictícias (todas 1.0)
                confiancas_ficticias = np.ones(len(boxes_para_nms), dtype=np.float32)

                # Limiar de sobreposição (IoU) - quanto menor, mais rigoroso
                nms_threshold = 0.3
                # Limiar de score baixo, pois não temos scores reais do MediaPipe aqui
                score_threshold = 0.1
                indices_mantidos = cv2.dnn.NMSBoxes(boxes_para_nms.tolist(), confiancas_ficticias.tolist(), score_threshold=score_threshold, nms_threshold=nms_threshold)

                if len(indices_mantidos) > 0:
                     # Achatamento pode ser necessário
                     if isinstance(indices_mantidos, tuple):
                         indices_finais = indices_mantidos[0] if len(indices_mantidos) > 0 else []
                     else:
                         indices_finais = indices_mantidos.flatten()

                     numero_rostos = len(indices_finais)
                     st.success(f"**Número final de rostos detectados: {numero_rostos}**")

                     # Desenhar apenas as boxes mantidas pelo NMS
                     for i in indices_finais:
                         # Usa a detecção válida original (x, y, w, h) correspondente ao índice
                         x, y, w, h = deteccoes_validas[i]
                         # Desenha o retângulo na imagem BGR de desenho
                         cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (36, 255, 12), 3) # Verde brilhante, espesso
                else:
                     st.info("**Nenhum rosto detectado após filtro de sobreposição (NMS).**")
                     numero_rostos = 0 # Garante que a contagem seja 0
        else:
            st.info("**Nenhum rosto detectado na imagem (em nenhuma rotação).**")

        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        st.error(f"Erro inesperado durante o processamento da imagem: {e}")
        st.error(f"Traceback: {traceback.format_exc()}") # Log detalhado para debug
        return -1, None # Indica erro

# --- Interface Gráfica com Streamlit ---

# Configurações da página (deve ser a primeira chamada do Streamlit)
st.set_page_config(page_title="Detector Facial MediaPipe", layout="wide", initial_sidebar_state="auto")

# Título da Aplicação
st.title("👁️ Detector de Rostos com MediaPipe")
st.write("Faça o upload de uma imagem (mesmo com rostos rotacionados) e veja quantos são detectados!")
st.markdown("---") # Linha separadora

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
        # Usar um identificador único para o cache baseado no nome e tamanho do arquivo
        # Isso ajuda se o usuário enviar um arquivo com mesmo nome mas conteúdo diferente
        cache_key = f"{arquivo_imagem_enviado.name}-{arquivo_imagem_enviado.size}"
        imagem_pil = Image.open(arquivo_imagem_enviado)

        # Criar duas colunas para exibir as imagens lado a lado
        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            # Exibir a imagem original usando a largura do container
            st.image(imagem_pil, caption=f"Original: {arquivo_imagem_enviado.name}", use_container_width=True)

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            # Mostrar um spinner enquanto processa
            with st.spinner('Detectando rostos (testando rotações)... Aguarde!'):
                # Chamar a função de detecção (passando a imagem PIL)
                # O cache @st.cache_data cuidará de não reprocessar se a imagem_pil for a mesma
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)

            # Se o processamento foi bem-sucedido (retornou uma imagem)
            if imagem_final_bgr is not None and num_rostos >= 0:
                # Converter a imagem processada de BGR (OpenCV) para RGB (Streamlit/PIL)
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                # Exibir a imagem processada usando a largura do container
                st.image(imagem_final_rgb, caption=f"Processada: {num_rostos} rosto(s) detectado(s)", use_container_width=True)

                # Oferecer botão para download da imagem processada
                # Converter array NumPy BGR para bytes PNG para download
                sucesso_encode, buffer = cv2.imencode('.png', imagem_final_bgr)
                if sucesso_encode:
                    img_bytes = buffer.tobytes()
                    # Extrai o nome base do arquivo original para usar no download
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
                 # Mensagem se nenhum rosto foi detectado, mas o processamento ocorreu sem erro
                 # Não precisa mostrar a imagem original aqui de novo, pois ela já está na outra coluna
                 st.info("Processamento concluído, mas nenhum rosto foi detectado nesta imagem.")
            else:
                # Mensagem se o processamento falhou (num_rostos == -1 ou imagem_final_bgr is None)
                st.error("Falha ao processar a imagem. Verifique os logs se disponíveis.")

    except Exception as e:
        # Captura erros ao tentar abrir a imagem (ex: arquivo corrompido, formato inválido)
        st.error(f"Erro ao carregar ou processar o arquivo de imagem: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    # Mensagem inicial quando nenhum arquivo foi enviado ainda
    st.info("☝️ Faça o upload de uma imagem para iniciar a detecção.")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")
