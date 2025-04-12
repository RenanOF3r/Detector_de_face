# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import traceback # Para logs de erro detalhados
import time # Para timestamps nos logs

# --- Configuração do MediaPipe (fora da função para eficiência) ---
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    print(f"[{time.time()}] Detector MediaPipe inicializado.", flush=True)
except AttributeError as e:
    print(f"[{time.time()}] ERRO CRÍTICO: Falha ao inicializar MediaPipe (AttributeError): {e}", flush=True)
    st.error(f"Erro Crítico: Falha ao inicializar componentes do MediaPipe. Verifique a instalação. Detalhes: {e}")
    st.stop()
except Exception as e:
    print(f"[{time.time()}] ERRO CRÍTICO: Falha ao inicializar MediaPipe (Exception): {e}", flush=True)
    st.error(f"Erro Crítico: Erro inesperado ao inicializar o MediaPipe. Detalhes: {e}")
    st.stop()

# --- Função de Rotação ---
def rotacionar_imagem(imagem_np, angulo):
    """Rotaciona uma imagem NumPy (BGR ou RGB)."""
    if angulo == 0: return imagem_np
    elif angulo == 90: return cv2.rotate(imagem_np, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180: return cv2.rotate(imagem_np, cv2.ROTATE_180)
    elif angulo == 270: return cv2.rotate(imagem_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return imagem_np

# --- Função de Transformação de Coordenadas (com Logs) ---
def transformar_coordenadas_box(box_rotada, angulo_rotacao, dim_originais, dim_rotadas):
    """
    Transforma as coordenadas de uma bounding box detectada na imagem rotacionada
    de volta para o sistema de coordenadas da imagem original. Retorna inteiros.
    """
    print(f"--- Iniciando transformação ---", flush=True)
    print(f"  Box Rotada (x,y,w,h): {box_rotada}", flush=True)
    print(f"  Ângulo: {angulo_rotacao}", flush=True)
    print(f"  Dim Originais (h,w): {dim_originais}", flush=True)
    print(f"  Dim Rotadas (h,w): {dim_rotadas}", flush=True)
    try:
        xmin_r, ymin_r, w_r, h_r = box_rotada
        h_orig, w_orig = dim_originais
        h_rot, w_rot = dim_rotadas

        if w_orig <= 0 or h_orig <= 0 or w_rot <= 0 or h_rot <= 0:
            print(f"  ERRO Transformação: Dimensões inválidas (orig={dim_originais}, rot={dim_rotadas})", flush=True)
            return None

        xmax_r = xmin_r + w_r
        ymax_r = ymin_r + h_r
        xmin_r = max(0, xmin_r)
        ymin_r = max(0, ymin_r)
        xmax_r = min(w_rot, xmax_r)
        ymax_r = min(h_rot, ymax_r)
        print(f"  Box Rotada Ajustada (x1,y1,x2,y2): ({xmin_r}, {ymin_r}, {xmax_r}, {ymax_r})", flush=True)

        if angulo_rotacao == 0: xmin_o, ymin_o, xmax_o, ymax_o = xmin_r, ymin_r, xmax_r, ymax_r
        elif angulo_rotacao == 90: xmin_o, ymin_o, xmax_o, ymax_o = ymin_r, w_rot - xmax_r, ymax_r, w_rot - xmin_r
        elif angulo_rotacao == 180: xmin_o, ymin_o, xmax_o, ymax_o = w_rot - xmax_r, h_rot - ymax_r, w_rot - xmin_r, h_rot - ymin_r
        elif angulo_rotacao == 270: xmin_o, ymin_o, xmax_o, ymax_o = h_rot - ymax_r, xmin_r, h_rot - ymin_r, xmax_r
        else:
            print(f"  ERRO Transformação: Ângulo inválido {angulo_rotacao}", flush=True)
            return None
        print(f"  Coords Originais Brutas (x1,y1,x2,y2): ({xmin_o:.2f}, {ymin_o:.2f}, {xmax_o:.2f}, {ymax_o:.2f})", flush=True)

        xmin_o, ymin_o, xmax_o, ymax_o = int(round(xmin_o)), int(round(ymin_o)), int(round(xmax_o)), int(round(ymax_o))
        print(f"  Coords Originais Arredondadas (x1,y1,x2,y2): ({xmin_o}, {ymin_o}, {xmax_o}, {ymax_o})", flush=True)

        w_o_pre = max(0, xmax_o - xmin_o)
        h_o_pre = max(0, ymax_o - ymin_o)
        print(f"  Dimensões Originais Pré-ajuste (w,h): ({w_o_pre}, {h_o_pre})", flush=True)

        xmin_o = max(0, xmin_o)
        ymin_o = max(0, ymin_o)
        xmin_o = min(w_orig - 1, xmin_o) if w_orig > 0 else 0
        ymin_o = min(h_orig - 1, ymin_o) if h_orig > 0 else 0
        print(f"  Coords Originais Pós-ajuste (x1,y1): ({xmin_o}, {ymin_o})", flush=True)

        w_o = max(0, xmax_o - xmin_o)
        h_o = max(0, ymax_o - ymin_o)
        w_o = min(w_orig - xmin_o, w_o) if w_orig > xmin_o else 0
        h_o = min(h_orig - ymin_o, h_o) if h_orig > ymin_o else 0
        print(f"  Dimensões Originais Finais (w,h): ({w_o}, {h_o})", flush=True)

        if w_o > 0 and h_o > 0:
            resultado_final = (xmin_o, ymin_o, w_o, h_o)
            print(f"  SUCESSO Transformação: Retornando {resultado_final}", flush=True)
            return resultado_final
        else:
            print(f"  FALHA Transformação: Dimensões finais inválidas (w={w_o}, h={h_o})", flush=True)
            return None
    except Exception as e:
        print(f"  ERRO Transformação (Exceção): {e}\n{traceback.format_exc()}", flush=True)
        st.warning(f"Aviso: Erro durante a transformação de coordenadas: {e}")
        return None


# --- Função de Processamento Principal (com Logs e Indentação Corrigida) ---
@st.cache_data(show_spinner=False)
def detectar_e_desenhar_rostos(_imagem_pil):
    """
    Detecta rostos em uma imagem PIL, tentando rotações, aplica NMS e retorna
    a contagem e a imagem processada (como array NumPy BGR).
    O argumento _imagem_pil é ignorado pelo cache do Streamlit.
    """
    print(f"\n[{time.time()}] ===== INICIANDO DETECÇÃO =====", flush=True)
    try:
        imagem_rgb_original = np.array(_imagem_pil.convert('RGB'))
        imagem_rgb_original = np.copy(imagem_rgb_original)
        imagem_rgb_original.flags.writeable = True

        h_orig, w_orig, _ = imagem_rgb_original.shape
        print(f"[{time.time()}] Dimensões Originais (H, W): ({h_orig}, {w_orig})", flush=True)
        if h_orig == 0 or w_orig == 0:
            print(f"[{time.time()}] ERRO: Imagem com dimensões inválidas.", flush=True)
            st.error("Erro: Imagem carregada tem dimensões inválidas.")
            return 0, None

        imagem_bgr_para_desenho = cv2.cvtColor(imagem_rgb_original, cv2.COLOR_RGB2BGR)
        deteccoes_finais_orig = [] # Lista para guardar boxes (x, y, w, h) nas coordenadas originais

        # Iterar sobre as rotações
        for angulo in [0, 90, 180, 270]:
            print(f"\n[{time.time()}] --- Processando Ângulo: {angulo} graus ---", flush=True)
            imagem_rgb_rotacionada = rotacionar_imagem(imagem_rgb_original, angulo)
            imagem_rgb_rotacionada = np.copy(imagem_rgb_rotacionada)
            imagem_rgb_rotacionada.flags.writeable = True
            h_rot, w_rot, _ = imagem_rgb_rotacionada.shape
            print(f"[{time.time()}] Dimensões Rotacionadas (H, W): ({h_rot}, {w_rot})", flush=True)
            if h_rot == 0 or w_rot == 0:
                print(f"[{time.time()}] AVISO: Imagem rotacionada inválida para ângulo {angulo}.", flush=True)
                st.warning(f"Aviso: Imagem rotacionada para {angulo} graus tem dimensões inválidas.")
                continue

            # Processar a imagem rotacionada com o MediaPipe
            print(f"[{time.time()}] Enviando imagem rotacionada para MediaPipe...", flush=True)
            resultados = face_detector.process(imagem_rgb_rotacionada)
            print(f"[{time.time()}] MediaPipe processou.", flush=True)

            if resultados.detections:
                print(f"[{time.time()}] DETECÇÕES ENCONTRADAS no ângulo {angulo}: {len(resultados.detections)}", flush=True)
                # Processar cada detecção encontrada nesta rotação
                for i, detection in enumerate(resultados.detections):
                    print(f"\n  Processando Detecção #{i+1} (Ângulo {angulo})", flush=True)
                    box_relativa = detection.location_data.relative_bounding_box
                    score = detection.score[0] if detection.score else 'N/A'
                    print(f"    Score: {score}", flush=True)
                    if box_relativa:
                        print(f"    Box Relativa (xmin,ymin,w,h): ({box_relativa.xmin:.4f}, {box_relativa.ymin:.4f}, {box_relativa.width:.4f}, {box_relativa.height:.4f})", flush=True)
                        # Verifica validade básica da box relativa (INDENTAÇÃO CORRIGIDA AQUI E ABAIXO)
                        if not (0 <= box_relativa.xmin <= 1 and 0 <= box_relativa.ymin <= 1 and box_relativa.width > 0 and box_relativa.height > 0):
                            print(f"    AVISO: Box relativa inválida, pulando.", flush=True)
                            continue # Ignora box relativa inválida

                        # Converter para coordenadas de pixel ABSOLUTAS na imagem ROTACIONADA
                        xmin_r = math.floor(box_relativa.xmin * w_rot)
                        ymin_r = math.floor(box_relativa.ymin * h_rot)
                        w_r = math.ceil(box_relativa.width * w_rot)
                        h_r = math.ceil(box_relativa.height * h_rot)
                        box_abs_rotada = (xmin_r, ymin_r, w_r, h_r)
                        print(f"    Box Absoluta Rotacionada (x,y,w,h): {box_abs_rotada}", flush=True)

                        # Transformar as coordenadas de volta para a imagem ORIGINAL
                        box_abs_original = transformar_coordenadas_box(
                            box_abs_rotada, angulo, (h_orig, w_orig), (h_rot, w_rot)
                        )

                        if box_abs_original: # Verifica se a transformação retornou uma box válida
                            print(f"    SUCESSO: Box transformada para coords originais: {box_abs_original}", flush=True)
                            deteccoes_finais_orig.append(box_abs_original)
                        else:
                            print(f"    FALHA: Transformação da box falhou ou retornou None.", flush=True)
                    else:
                        print(f"    AVISO: Detecção sem location_data.relative_bounding_box.", flush=True)
            else:
                print(f"[{time.time()}] Nenhuma detecção encontrada por MediaPipe no ângulo {angulo}.", flush=True)

        # --- Pós-processamento: Desenhar TODAS as detecções (SEM NMS para teste) ---
        print(f"\n[{time.time()}] --- Pós-processamento (NMS Desabilitado) ---", flush=True)
        print(f"[{time.time()}] Total de detecções transformadas com sucesso: {len(deteccoes_finais_orig)}", flush=True)
        numero_rostos = 0
        if deteccoes_finais_orig:
            # Filtra novamente por segurança, garantindo tuplas de 4 inteiros positivos
            deteccoes_validas = [d for d in deteccoes_finais_orig if isinstance(d, tuple) and len(d) == 4 and all(isinstance(v, int) and v >= 0 for v in d) and d[2] > 0 and d[3] > 0]
            print(f"[{time.time()}] Detecções válidas após filtro final: {len(deteccoes_validas)}", flush=True)

            if not deteccoes_validas:
                st.info("Nenhuma detecção válida encontrada após transformações.")
                print(f"[{time.time()}] Nenhuma detecção válida após filtro final.", flush=True)
                numero_rostos = 0
            else:
                numero_rostos = len(deteccoes_validas)
                st.warning(f"**NMS Desabilitado (Teste): {numero_rostos} detecção(ões) encontrada(s) antes do filtro.**")
                print(f"[{time.time()}] Desenhando {numero_rostos} caixas (NMS desabilitado).", flush=True)

                # Desenhar TODAS as caixas válidas encontradas
                for i in range(len(deteccoes_validas)):
                    x, y, w, h = deteccoes_validas[i]
                    print(f"  Desenhando caixa {i+1}: ({x}, {y}, {w}, {h})", flush=True)
                    cv2.rectangle(imagem_bgr_para_desenho, (x, y), (x + w, y + h), (255, 0, 0), 2) # Azul, espessura 2
        else:
            st.info("**Nenhum rosto detectado na imagem (em nenhuma rotação) ou falha na transformação.**")
            print(f"[{time.time()}] Nenhuma detecção inicial sobreviveu à transformação.", flush=True)
            numero_rostos = 0

        print(f"[{time.time()}] ===== DETECÇÃO FINALIZADA =====", flush=True)
        return numero_rostos, imagem_bgr_para_desenho
        # --- FIM DO BLOCO MODIFICADO ---

    except Exception as e:
        print(f"[{time.time()}] ERRO INESPERADO em detectar_e_desenhar_rostos: {e}\n{traceback.format_exc()}", flush=True)
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
    print(f"\n[{time.time()}] Upload de arquivo detectado: {arquivo_imagem_enviado.name}", flush=True)
    try:
        imagem_pil = Image.open(arquivo_imagem_enviado)
        print(f"[{time.time()}] Imagem PIL carregada com sucesso.", flush=True)
        coluna_original, coluna_processada = st.columns(2)

        with coluna_original:
            st.subheader("🖼️ Imagem Original")
            st.image(imagem_pil, caption=f"Original: {arquivo_imagem_enviado.name}", use_container_width=True)

        with coluna_processada:
            st.subheader("✨ Imagem Processada")
            with st.spinner('Detectando rostos (testando rotações)... Aguarde!'):
                print(f"[{time.time()}] Chamando detectar_e_desenhar_rostos...", flush=True)
                # Chama a função (agora com logs e sem NMS ativo internamente)
                num_rostos, imagem_final_bgr = detectar_e_desenhar_rostos(imagem_pil)
                print(f"[{time.time()}] Retornou de detectar_e_desenhar_rostos.", flush=True)


            if imagem_final_bgr is not None and num_rostos >= 0:
                print(f"[{time.time()}] Processamento OK. Rosto(s) detectado(s) (antes NMS): {num_rostos}", flush=True)
                imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)
                st.image(imagem_final_rgb, caption=f"Processada (NMS Desabilitado): {num_rostos} detecção(ões)", use_container_width=True)
                sucesso_encode, buffer = cv2.imencode('.png', imagem_final_bgr)
                if sucesso_encode:
                    img_bytes = buffer.tobytes()
                    nome_base = arquivo_imagem_enviado.name.rsplit('.', 1)[0]
                    st.download_button(
                       label="💾 Baixar Imagem (Teste NMS Desabilitado)",
                       data=img_bytes,
                       file_name=f"rostos_detectados_sem_nms_{nome_base}.png",
                       mime="image/png"
                    )
                else:
                    print(f"[{time.time()}] AVISO: Falha ao encodar imagem para download.", flush=True)
                    st.warning("Não foi possível gerar o arquivo para download.")
            elif num_rostos == 0:
                 print(f"[{time.time()}] Processamento OK, mas 0 rostos detectados/válidos.", flush=True)
                 st.info("Processamento concluído, mas nenhuma detecção válida foi encontrada nesta imagem (antes do NMS).")
            else: # num_rostos == -1 indica erro na função
                print(f"[{time.time()}] ERRO: Função de detecção retornou erro (num_rostos = -1).", flush=True)
                st.error("Falha ao processar a imagem. Verifique os logs se disponíveis.")

    except Exception as e:
        print(f"[{time.time()}] ERRO ao carregar/processar imagem no Streamlit: {e}\n{traceback.format_exc()}", flush=True)
        st.error(f"Erro ao carregar ou processar o arquivo de imagem: {e}")
        st.warning("Verifique se o arquivo enviado é uma imagem válida e tente novamente.")

else:
    # Mensagem inicial quando nenhum arquivo foi enviado ainda
    st.info("☝️ Faça o upload de uma imagem para iniciar a detecção.")

st.markdown("---")
st.markdown("Desenvolvido com [Streamlit](https://streamlit.io/) & [Google MediaPipe](https://developers.google.com/mediapipe)")
