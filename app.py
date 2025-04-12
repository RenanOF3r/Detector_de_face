# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image # Streamlit trabalha bem com PIL

# --- Configuração do MediaPipe (fora da função para eficiência) ---
# Inicializa a solução de detecção de rosto
mp_face_detection = mp.solutions.face_detection
# Inicializa utilitários para desenho
mp_drawing = mp.solutions.drawing_utils
# Configura o detector:
# model_selection=0: para rostos próximos (até 2m)
# model_selection=1: para rostos mais distantes (até 5m)
# min_detection_confidence: limiar de confiança para considerar uma detecção válida
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- Função de Processamento ---
def detectar_e_desenhar_rostos(imagem_pil):
    """
    Detecta rostos em uma imagem PIL usando MediaPipe e retorna a contagem
    e a imagem processada (como array NumPy BGR).

    Args:
        imagem_pil (PIL.Image.Image): A imagem carregada pelo usuário.

    Returns:
        tuple: (numero_rostos, imagem_processada_bgr) ou (None, None) em caso de erro.
    """
    try:
        # Converter a imagem PIL para um array NumPy no formato RGB
        imagem_rgb = np.array(imagem_pil.convert('RGB'))

        # Processar a imagem com o MediaPipe (espera RGB)
        resultados = face_detector.process(imagem_rgb)

        # Criar uma cópia da imagem original (em BGR para desenho com OpenCV)
        # Convertendo de RGB (numpy) para BGR (OpenCV)
        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)

        numero_rostos = 0
        if resultados.detections:
            numero_rostos = len(resultados.detections)
            st.success(f"**Número de rostos detectados: {numero_rostos}**") # Mensagem de sucesso

            # Desenhar as caixas delimitadoras e pontos chave na cópia BGR
            for detection in resultados.detections:
                mp_drawing.draw_detection(imagem_bgr_para_desenho, detection)
        else:
            st.info("**Nenhum rosto detectado na imagem.**") # Mensagem informativa

        return numero_rostos, imagem_bgr_para_desenho

    except Exception as e:
        st.error(f"Erro durante o processamento da imagem com MediaPipe: {e}")
        return None, None

# --- Interface Gráfica com Streamlit ---

# Configurações da página
st.set_page_config(page_title="Detector Facial MediaPipe", layout="wide", initial_sidebar_state="collapsed")

# Título da Aplicação
st.title("👁️ Detector de Rostos com MediaPipe")
st.write("Faça o upload de uma imagem e veja quantos rostos são detectados!")

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
            # Exibir a imagem original
            st.image(imagem_pil, caption="Imagem enviada", use_column_width='always')

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            # Mostrar um spinner enquanto processa
            with st.spinner('Detectando rostos... Aguarde!'):
                # Chamar a função de detecção
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)

            # Se o processamento foi bem-sucedido
            if imagem_final_bgr is not None:
                # Converter a imagem processada de BGR (OpenCV) para RGB (Streamlit/PIL)
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                # Exibir a imagem processada
                st.image(imagem_final_rgb, caption=f"Detecção concluída ({num_rostos} rosto(s))", use_column_width='always')

                # Oferecer botão para download da imagem processada
                # Converter array NumPy RGB para bytes para download
                img_bytes = cv2.imencode('.png', imagem_final_bgr)[1].tobytes() # Salva como PNG em memória
                st.download_button(
                   label="Baixar Imagem Processada",
                   data=img_bytes,
                   file_name=f"rostos_detectados_{arquivo_imagem_enviado.name}.png",
                   mime="image/png"
                )
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
