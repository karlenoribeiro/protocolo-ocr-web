# -*- coding: utf-8 -*-
"""
Núcleo de OCR:
- Pré-processamentos em variações (CLAHE, Otsu, invertido, morfologia)
- Deskew usando OSD do Tesseract
- Regex robusta para formatos de protocolo
- Saída padronizada
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from unidecode import unidecode
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Belem")

# --------------------------------------------------------------------------------------
# Tempo e helpers
# --------------------------------------------------------------------------------------

def agora_belem_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def tesseract_config(digits_only: bool = False) -> str:
    # psm 6: assumindo um bloco único de texto
    if digits_only:
        return "--psm 6 -c tessedit_char_whitelist=0123456789"
    return "--psm 6"


def ocr_text(img_bgr: np.ndarray, config: str) -> str:
    try:
        txt = pytesseract.image_to_string(img_bgr, config=config, lang=_langs())
        # normaliza acentos e espaços
        return unidecode(txt).replace("\r", " ").replace("\n", " ").strip()
    except Exception:
        return ""


def _langs() -> str:
    """
    Usa 'por+eng' se disponível, senão volta para 'eng'.
    """
    try:
        langs = set(pytesseract.get_languages(config=""))
        if "por" in langs:
            return "por+eng"
    except Exception:
        pass
    return "eng"

# --------------------------------------------------------------------------------------
# Pré-processamento / deskew
# --------------------------------------------------------------------------------------

def deskew_osd(bgr: np.ndarray) -> np.ndarray:
    """
    Estima rotação via OSD do Tesseract e corrige.
    """
    try:
        osd = pytesseract.image_to_osd(bgr)
        angle = 0.0
        for part in osd.split():
            try:
                angle = float(part)
                break
            except ValueError:
                continue
        if abs(angle) > 0.01:
            (h, w) = bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return bgr


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def clahe(gray: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return c.apply(gray)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def morph_open(bin_img: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)


def tentar_ocr_em_varias_formas(bgr: np.ndarray) -> str:
    """
    Testa variações de pré-processamento para maximizar o OCR.
    """
    candidates = []

    # 0) Original (deskew)
    img0 = deskew_osd(bgr)
    candidates.append(img0)

    # 1) Gray + CLAHE
    g1 = clahe(to_gray(img0))
    candidates.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # 2) Otsu
    b1 = binarize_otsu(g1)
    candidates.append(cv2.cvtColor(b1, cv2.COLOR_GRAY2BGR))

    # 3) Invertido
    b2 = cv2.bitwise_not(b1)
    candidates.append(cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR))

    # 4) Morfologia
    b3 = morph_open(b1, 3)
    candidates.append(cv2.cvtColor(b3, cv2.COLOR_GRAY2BGR))

    # 5) Blur leve
    g2 = cv2.GaussianBlur(g1, (3, 3), 0)
    candidates.append(cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR))

    # OCR nas variantes
    texts = []
    for c in candidates:
        t = ocr_text(c, tesseract_config(digits_only=False))
        if t:
            texts.append(t)

    # Concatena tudo (evita perder algo que só apareceu em uma variante)
    return " | ".join(texts)


# --------------------------------------------------------------------------------------
# Extração de protocolo
# --------------------------------------------------------------------------------------

# Aceita:
# 23073-009780/2013-20
# 23073009780/2013-20 (com / e - opcionais, espaços)
PROTO_REGEXES = [
    re.compile(
        r'PROTOCOLO[:\s-]*('
        r'[0-9]{5}\s*[-–]?\s*[0-9]{5,6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2}'
        r')',
        re.IGNORECASE
    ),
    re.compile(
        r'('
        r'[0-9]{5}\s*[-–]?\s*[0-9]{5,6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2}'
        r')'
    ),
]


def extrair_protocolo_de_texto(texto: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Procura um protocolo "formatado" e devolve também a versão só com 17 dígitos.
    Retorna (protocolo_formatado, protocolo_17_digitos).
    """
    if not texto:
        return (None, None)

    for rx in PROTO_REGEXES:
        m = rx.search(texto)
        if m:
            fmt = m.group(1)
            # remove tudo que não é dígito
            digits = "".join(ch for ch in fmt if ch.isdigit())
            if len(digits) == 17:
                return (fmt.strip(), digits)

    # fallback: às vezes o OCR junta tudo em um bloco de 17 dígitos
    m2 = re.search(r'(\d{17})', texto)
    if m2:
        dg = m2.group(1)
        return (dg, dg)

    return (None, None)

# --------------------------------------------------------------------------------------
# ROI (opcional)
# --------------------------------------------------------------------------------------

def find_protocolo_roi(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Heurística simples (placeholder): retorna None para usar a imagem toda.
    Se quiser restringir a área (ex.: cabeçalho), implemente aqui.
    """
    return None


# --------------------------------------------------------------------------------------
# Função pública usada pelo backend
# --------------------------------------------------------------------------------------

def processar_imagem_bytes(filename: str, content: bytes) -> Dict[str, Any]:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "arquivo": filename,
            "protocolo_ocr": None,
            "protocolo_17_digitos": None,
            "data_extracao": agora_belem_str(),
            "ok": False,
        }

    textos = ""
    roi = find_protocolo_roi(img)
    if roi:
        x1, y1, x2, y2 = roi
        recorte = img[y1:y2, x1:x2]
        textos = tentar_ocr_em_varias_formas(recorte)

    if not textos.strip():
        textos = tentar_ocr_em_varias_formas(img)

    protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(textos)

    # Tentativa extra, caso não tenha achado
    if not protocolo_17:
        texto_extra = ocr_text(img, tesseract_config(digits_only=False))
        if texto_extra:
            protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(texto_extra)

    ok = bool(protocolo_17)
    return {
        "arquivo": filename,
        "protocolo_ocr": protocolo_fmt,
        "protocolo_17_digitos": protocolo_17,
        "data_extracao": agora_belem_str(),
        "ok": ok,
    }
