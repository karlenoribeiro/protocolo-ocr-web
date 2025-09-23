# =======================
# EXTRAÇÃO DE PROTOCOLOS (17 DÍGITOS) DE IMAGENS
# Mantém a lógica original, adicionando o campo "ok" no resultado por arquivo.
# =======================

import re  # Módulo para trabalhar com expressões regulares (regex)
import cv2  # Biblioteca OpenCV para processamento de imagens
import numpy as np  # Biblioteca NumPy para manipulação de arrays/matrizes
import pandas as pd  # Biblioteca Pandas para manipulação de dados em formato tabular
import pytesseract  # Interface Python para o OCR Tesseract
from unidecode import unidecode  # Remove acentos/caracteres especiais de strings
from datetime import datetime  # Trabalha com datas e horários
from zoneinfo import ZoneInfo  # Define zonas de fuso horário
from pathlib import Path  # Manipulação de caminhos de arquivos
from typing import Optional, Tuple, List  # Tipagem para anotações de funções
from openpyxl import load_workbook  # Manipulação de arquivos Excel

# Define o fuso horário de Belém para padronizar os horários gerados
LOCAL_TZ = ZoneInfo("America/Belem")

# Extensões de imagem aceitas pelo sistema
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jfif"}

# Função que retorna a data e hora atuais no formato string ajustado para o fuso horário definido
def now_belem_str():
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

# Detecta os idiomas suportados pelo Tesseract no sistema
def _langs():
    try:
        langs = set(pytesseract.get_languages(config=""))  # Lista idiomas disponíveis
        return "por+eng" if "por" in langs else "eng"  # Retorna português+inglês se disponível, senão só inglês
    except Exception:
        return "por+eng"  # Caso ocorra erro, define padrão como português+inglês

# Define configuração global de idiomas a serem usados no OCR
OCR_LANGS = _langs()

# Converte imagem colorida para escala de cinza
def to_gray(img):
    if len(img.shape) == 3:  # Verifica se a imagem tem 3 canais (colorida)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte para cinza
    return img  # Retorna inalterada se já estiver em tons de cinza

# Rotaciona imagem sem perder bordas, evitando cortes
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]  # Obtém altura e largura da imagem
    center = (w // 2, h // 2)  # Define o centro da imagem para rotação
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Matriz de rotação
    cos = np.abs(M[0, 0]); sin = np.abs(M[1, 0])  # Cálculo do cosseno e seno do ângulo
    nW = int((h * sin) + (w * cos))  # Nova largura após rotação
    nH = int((h * cos) + (w * sin))  # Nova altura após rotação
    M[0, 2] += (nW / 2) - center[0]  # Ajuste horizontal
    M[1, 2] += (nH / 2) - center[1]  # Ajuste vertical
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)  # Aplica rotação

# Função para detectar e corrigir rotação automática de imagens usando OSD do Tesseract
def deskew_osd(img):
    try:
        osd = pytesseract.image_to_osd(img)  # OCR detecta informações de orientação
        angle_m = re.search(r"Rotate: (\d+)", osd)  # Captura o ângulo
        if angle_m:
            angle = float(angle_m.group(1)) % 360  # Normaliza ângulo
            if angle > 1:  # Só rotaciona se ângulo for maior que 1 grau
                return rotate_bound(img, angle)
    except Exception:
        pass
    return img  # Retorna imagem original se falhar

# Melhora qualidade da imagem para leitura OCR
def enhance_for_ocr(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Equalização de histograma
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3,3), 0)  # Suaviza ruídos
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Binarização adaptativa
    return th

# Configuração padrão do Tesseract OCR
def tesseract_config(digits_only=False):
    base = f"--oem 3 --psm 6 -l {OCR_LANGS}"  # Define modo de OCR e idiomas
    if digits_only:
        base += " -c tessedit_char_whitelist=0123456789-/"  # Limita caracteres a dígitos e separadores
    return base

# Função para realizar OCR e extrair texto completo
def ocr_text(img, config):
    return pytesseract.image_to_string(img, config=config)

# Função para realizar OCR e retornar dados estruturados como DataFrame
def ocr_data(img, config):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=config)

# Localiza a região onde a palavra "PROTOCOLO" aparece na imagem
def find_protocolo_roi(img) -> Optional[Tuple[int,int,int,int]]:
    df = ocr_data(img, tesseract_config(digits_only=False))  # Executa OCR completo
    try:
        df = df.dropna(subset=['text'])  # Remove linhas vazias
    except Exception:
        return None
    if df is None or len(df)==0:
        return None
    df['norm'] = df['text'].astype(str).apply(lambda s: unidecode(s).upper())  # Normaliza texto
    mask = df['norm'].str.contains(r'^PROTOC', regex=True)  # Filtra palavra "PROTOC"
    cand = df[mask]
    if cand.empty:
        return None
    cand = cand.sort_values(['top','left']).iloc[0]  # Seleciona primeiro resultado encontrado
    x, y, w, h = int(cand['left']), int(cand['top']), int(cand['width']), int(cand['height'])
    H, W = img.shape[:2]
    pad_y = int(h*2.5)  # Margem vertical
    pad_x_left = int(w*0.5)  # Margem lateral esquerda
    x1 = max(0, x - pad_x_left)
    y1 = max(0, y - pad_y//2)
    x2 = W
    y2 = min(H, y + pad_y)
    return (x1, y1, x2, y2)  # Retorna coordenadas do recorte

# Padrões de regex para identificar protocolo (formato 17 dígitos)
PROTO_REGEXES = [
    re.compile(r'PROTOCOLO[:\s-]*([0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2})', re.IGNORECASE),
    re.compile(r'([0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2})')
]

# Limpa caracteres que não sejam dígitos, garantindo apenas 17 números
def limpar_para_17_digitos(seq: str) -> Optional[str]:
    if not seq:
        return None
    digits = re.sub(r'\D', '', seq)  # Remove não dígitos
    return digits if len(digits) == 17 else None  # Valida tamanho

# Extrai protocolo válido do texto bruto
def extrair_protocolo_de_texto(texto: str):
    if not texto:
        return (None, None)
    for rgx in PROTO_REGEXES:  # Testa cada padrão regex
        m = rgx.search(texto)
        if m:
            bruto = m.group(1)
            apenas_numeros = limpar_para_17_digitos(bruto)
            if apenas_numeros:
                # Formata no padrão XXXXX-XXXXXX/XXXX-XX
                fmt = f"{apenas_numeros[0:5]}-{apenas_numeros[5:11]}/{apenas_numeros[11:15]}-{apenas_numeros[15:17]}"
                return (fmt, apenas_numeros)
            return (bruto, None)
    return (None, None)

# Aplica várias técnicas de OCR para tentar extrair o texto
def tentar_ocr_em_varias_formas(img_color):
    gray1 = to_gray(deskew_osd(img_color))  # Converte e corrige rotação
    th = enhance_for_ocr(gray1)  # Melhora imagem
    inv = cv2.bitwise_not(th)  # Inverte cores
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)  # Operação morfológica
    cfg = tesseract_config(digits_only=True)
    textos = []
    for candidate in [gray1, th, inv, morph]:  # Testa cada variação
        try:
            textos.append(ocr_text(candidate, cfg))
        except Exception:
            pass
    return "\n".join([t for t in textos if t])  # Junta resultados

# Processa uma única imagem recebida em bytes
def processar_imagem_bytes(filename: str, content: bytes):
    arr = np.frombuffer(content, dtype=np.uint8)  # Converte bytes para array NumPy
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Decodifica em imagem colorida
    if img is None:
        return {  # Retorna resultado vazio se falhar
            "arquivo": filename,
            "protocolo_ocr": None,
            "protocolo_17_digitos": None,
            "data_extracao": now_belem_str(),
            "ok": False
        }

    roi = find_protocolo_roi(img)  # Localiza região de interesse
    textos = ""
    if roi:
        x1,y1,x2,y2 = roi
        recorte = img[y1:y2, x1:x2]  # Recorta imagem na ROI
        textos = tentar_ocr_em_varias_formas(recorte)

    if not textos.strip():  # Caso não encontre na ROI, tenta na imagem inteira
        textos = tentar_ocr_em_varias_formas(img)

    protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(textos)

    if not protocolo_17:  # Tenta OCR completo caso ainda não encontre
        try:
            texto_extra = ocr_text(img, tesseract_config(digits_only=False))
            protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(texto_extra)
        except Exception:
            pass

    ok = bool(protocolo_17)  # Marca como verdadeiro se encontrou protocolo
    return {
        "arquivo": filename,
        "protocolo_ocr": protocolo_fmt,
        "protocolo_17_digitos": protocolo_17,
        "data_extracao": now_belem_str(),
        "ok": ok
    }

# Caminho padrão para salvar os resultados em Excel
EXCEL_PATH = Path(__file__).parent / "data" / "protocolos_extraidos.xlsx"
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # Cria pasta se não existir

# Salva DataFrame em arquivo Excel, garantindo formatação adequada
def salvar_excel_texto(df_total: pd.DataFrame):
    if "protocolo_17_digitos" in df_total.columns:
        df_total["protocolo_17_digitos"] = df_total["protocolo_17_digitos"].fillna("").astype(str)
    else:
        df_total["protocolo_17_digitos"] = ""
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="w") as writer:
        df_total.to_excel(writer, index=False, sheet_name="Sheet1")
        ws = writer.book.active
        col_idx = None
        for idx, cell in enumerate(ws[1], start=1):
            if str(cell.value) == "protocolo_17_digitos":
                col_idx = idx
                break
        if col_idx is not None:  # Garante que a coluna seja tratada como texto
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.number_format = "@"

# Consolida resultados novos com existentes, removendo duplicados
def consolidar_resultados(novos: List[dict]) -> pd.DataFrame:
    # Cria DataFrame com estrutura padrão
    df_novos = pd.DataFrame(novos, columns=["arquivo","protocolo_ocr","protocolo_17_digitos","data_extracao"])
    if EXCEL_PATH.exists():
        try:
            df_existente = pd.read_excel(EXCEL_PATH, dtype={"protocolo_17_digitos": str})
        except Exception:
            df_existente = pd.DataFrame(columns=df_novos.columns)
    else:
        df_existente = pd.DataFrame(columns=df_novos.columns)
    
    # Junta dados novos e antigos
    df_total = pd.concat([df_existente, df_novos], ignore_index=True)
    if "protocolo_17_digitos" in df_total.columns:
        df_total["protocolo_17_digitos"] = df_total["protocolo_17_digitos"].astype(str)
    
    # Remove linhas duplicadas
    df_total = df_total.drop_duplicates(subset=["protocolo_17_digitos","arquivo"], keep="first")
    
    # Salva resultados consolidados em Excel
    salvar_excel_texto(df_total)
    return df_total
