# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from unidecode import unidecode
from zoneinfo import ZoneInfo


# -----------------------------------------------------------------------------
# Fuso para timestamps exibidos/gravados
# -----------------------------------------------------------------------------
LOCAL_TZ = ZoneInfo("America/Belem")

# -----------------------------------------------------------------------------
# Descoberta do executável do Tesseract (Windows)
# - respeita variáveis TESSERACT_CMD / TESSERACT_PATH
# - tenta caminhos comuns
# -----------------------------------------------------------------------------
_tess_cmd_env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
if _tess_cmd_env and os.path.exists(_tess_cmd_env):
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd_env  # type: ignore[attr-defined]
else:
    _candidatos = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _p in _candidatos:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p  # type: ignore[attr-defined]
            break


# -----------------------------------------------------------------------------
# Regex de protocolos (17 dígitos no total, com tolerância a espaços/hífens)
# Exemplos aceitos:
#   00000 000000 / 2024 - 00
#   00000-000000/2024-00
#   00000000000202400 (sequência já “colada”)
# -----------------------------------------------------------------------------
PROTO_REGEXES = [
    # sequência de 17 dígitos "colada"
    re.compile(r"\b(\d{17})\b"),

    # padrão com 5 + 6 + 4 + 2 (com separadores variados)
    re.compile(
        r'('
        r'[0-9]{5}\s*[-–]?\s*[0-9]{5,6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2}'
        r')',
        re.IGNORECASE
    ),
]


def _only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def extrai_protocolo(texto: str) -> Optional[str]:
    """
    Procura um protocolo que, ao retirar não-dígitos, resulte em exatamente 17 dígitos.
    Retorna a string de 17 dígitos (somente números) ou None.
    """
    if not texto:
        return None

    for rx in PROTO_REGEXES:
        for m in rx.finditer(texto):
            grupo = m.group(1)
            digits = _only_digits(grupo)
            if len(digits) == 17:
                return digits

    # fallback: varre qualquer bloco que possa virar 17 dígitos
    for m in re.finditer(r"[0-9\-\s/]{10,}", texto):
        digits = _only_digits(m.group(0))
        if len(digits) == 17:
            return digits

    return None


def deskew_osd(bgr: np.ndarray) -> np.ndarray:
    """
    Usa OSD do Tesseract para estimar rotação e corrigir (quando possível).
    Não falha a pipeline se der erro.
    """
    try:
        osd_text = pytesseract.image_to_osd(bgr)
        # Padrões típicos: "Rotate: 90" ou "Orientation in degrees: 90"
        m = re.search(
            r"(?:Rotate|Orientation in degrees)\s*:\s*([+-]?\d+)",
            osd_text,
            re.IGNORECASE,
        )
        angle = float(m.group(1)) if m else 0.0
        if abs(angle) > 0.01:
            (h, w) = bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            return cv2.warpAffine(
                bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    except Exception:
        pass
    return bgr


def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Pré-processamento simples: escala de cinza, normalização, limiarização adaptativa.
    Mantemos em BGR na entrada para compatibilidade e convertendo internamente.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # equalização leve
    gray = cv2.equalizeHist(gray)

    # binarização adaptativa ajuda em variações de iluminação
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 9
    )
    # Volta a 3 canais para APIs que esperam BGR/RGB
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)


def _ocr_text(bgr: np.ndarray) -> str:
    """
    OCR com Tesseract (por + eng). Ajuste o PSM conforme o layout típico.
    """
    config = "--psm 6"  # linhas/blocks com algum layout
    try:
        txt = pytesseract.image_to_string(bgr, lang="por+eng", config=config)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract não encontrado. Instale-o e/ou defina TESSERACT_CMD com o caminho do executável."
        ) from e
    return txt or ""


def processar_imagem_bytes(img_bytes: bytes, nome_arquivo: str = "") -> Dict[str, Any]:
    """
    Pipeline completa para 1 imagem:
      - decodifica
      - deskew (OSD)
      - pré-processa
      - OCR
      - extrai protocolo
    Retorna um dicionário serializável.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {
                "ok": False,
                "arquivo": nome_arquivo,
                "erro": "Falha ao decodificar a imagem (formato não suportado?).",
            }

        # Deskew (tenta, mas não é obrigatório)
        bgr = deskew_osd(bgr)

        # Pré-processamento e OCR
        prep = _preprocess_for_ocr(bgr)
        texto = _ocr_text(prep)

        # Normalizações auxiliares para busca
        texto_clean = unidecode(texto).upper()

        protocolo = extrai_protocolo(texto_clean)

        return {
            "ok": True,
            "arquivo": nome_arquivo,
            "texto_ocr": texto,
            "texto_ocr_clean": texto_clean,
            "protocolo_17_digitos": protocolo,
            "warnings": [] if protocolo else ["protocolo_nao_detectado"],
        }
    except Exception as e:
        return {"ok": False, "arquivo": nome_arquivo, "erro": str(e)}


def agora_belem_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now(LOCAL_TZ).strftime(fmt)
