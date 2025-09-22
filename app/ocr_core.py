# ============================================================
# ocr_core.py — Núcleo de OCR para extração de protocolos (17 dígitos)
# - Retorna dict com: arquivo, protocolo_ocr (formatado), protocolo_17_digitos, data_extracao, ok
# - Compatível com main_web.py entregue (usa apenas processar_imagem_bytes e EXCEL_PATH)
# - Inclui consolidar_resultados apenas por compatibilidade com versões antigas
# ============================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import pytesseract
from unidecode import unidecode
from datetime import datetime
from zoneinfo import ZoneInfo

# -------------------------
# Configurações gerais
# -------------------------
LOCAL_TZ = ZoneInfo("America/Belem")
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jfif"}

def now_belem_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _langs() -> str:
    """Detecta idiomas instalados no Tesseract e escolhe por+eng quando possível."""
    try:
        langs = set(pytesseract.get_languages(config=""))
        return "por+eng" if "por" in langs else "eng"
    except Exception:
        # fallback neutro (a maioria das instalações no Windows tem 'eng')
        return "eng"

OCR_LANGS = _langs()


# -------------------------
# Utilitários de imagem
# -------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotação preservando conteúdo dentro do canvas."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[1, 0])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)


def deskew_osd(img: np.ndarray) -> np.ndarray:
    """Tenta detectar ângulo com OSD do Tesseract e corrigir."""
    try:
        osd = pytesseract.image_to_osd(img)
        m = re.search(r"Rotate:\s*([0-9]+)", osd)
        if m:
            angle = float(m.group(1)) % 360
            if angle > 1:
                return rotate_bound(img, angle)
    except Exception:
        pass
    return img


def enhance_for_ocr(gray: np.ndarray) -> np.ndarray:
    """CLAHE + blur leve + Otsu."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def tesseract_config(digits_only: bool = False) -> str:
    """
    PSM 6: assume um bloco uniforme de texto.
    Quando digits_only=True, whitelist inclui dígitos e separadores mais comuns,
    mas o pipeline faz extração por regex de qualquer forma.
    """
    base = f"--oem 3 --psm 6 -l {OCR_LANGS}"
    if digits_only:
        base += " -c tessedit_char_whitelist=0123456789-/–"
    return base


def ocr_text(img: np.ndarray, config: str) -> str:
    try:
        return pytesseract.image_to_string(img, config=config)
    except Exception:
        return ""


def ocr_data(img: np.ndarray, config: str) -> pd.DataFrame:
    try:
        return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=config)
    except Exception:
        return pd.DataFrame(columns=["text", "left", "top", "width", "height"])


# -------------------------
# Localização e regex do protocolo
# -------------------------
def find_protocolo_roi(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Procura palavra que comece com 'PROTOC' e recorta uma faixa à direita.
    Retorna (x1, y1, x2, y2) ou None.
    """
    df = ocr_data(img, tesseract_config(digits_only=False))
    if df is None or df.empty or "text" not in df.columns:
        return None

    df = df.dropna(subset=["text"])
    if df.empty:
        return None

    df["norm"] = df["text"].astype(str).apply(lambda s: unidecode(s).upper())
    cand = df[df["norm"].str.contains(r"^PROTOC", regex=True, na=False)]
    if cand.empty:
        return None

    cand = cand.sort_values(["top", "left"]).iloc[0]
    x, y, w, h = int(cand["left"]), int(cand["top"]), int(cand["width"]), int(cand["height"])
    H, W = img.shape[:2]

    pad_y = int(h * 2.5)
    pad_x_left = int(w * 0.5)
    x1 = max(0, x - pad_x_left)
    y1 = max(0, y - pad_y // 2)
    x2 = W
    y2 = min(H, y + pad_y)
    return (x1, y1, x2, y2)


# Aceita variações com hífen curto/longo, espaços e barra
PROTO_REGEXES = [
    re.compile(
        r'PROTOCOLO[:\s-]*('
        r'[0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2}'
        r')',
        re.IGNORECASE
    ),
    re.compile(
        r'('
        r'[0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2}'
        r')'
    ),
]


def limpar_para_17_digitos(seq: str) -> Optional[str]:
    if not seq:
        return None
    digits = re.sub(r"\D", "", seq)
    return digits if len(digits) == 17 else None


def extrair_protocolo_de_texto(texto: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Retorna (protocolo_formatado, protocolo_17digitos)
    Ex.: "23073-009780/2013-20" , "23073009780201320"
    """
    if not texto:
        return (None, None)

    for rgx in PROTO_REGEXES:
        m = rgx.search(texto)
        if m:
            bruto = m.group(1)
            apenas_numeros = limpar_para_17_digitos(bruto)
            if apenas_numeros:
                fmt = (
                    f"{apenas_numeros[0:5]}-{apenas_numeros[5:11]}/"
                    f"{apenas_numeros[11:15]}-{apenas_numeros[15:17]}"
                )
                return (fmt, apenas_numeros)
            # Achou padrão mas não fechou 17 dígitos
            return (bruto, None)

    return (None, None)


def tentar_ocr_em_varias_formas(img_color: np.ndarray) -> str:
    """Executa OCR em variações (deskew, binário, invertido, morfologia) e concatena resultados."""
    gray1 = to_gray(deskew_osd(img_color))
    th = enhance_for_ocr(gray1)
    inv = cv2.bitwise_not(th)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    cfg_digits = tesseract_config(digits_only=True)
    textos = []
    for candidate in (gray1, th, inv, morph):
        txt = ocr_text(candidate, cfg_digits)
        if txt:
            textos.append(txt)
    return "\n".join(textos)


# -------------------------
# Função principal usada pela API
# -------------------------
def processar_imagem_bytes(filename: str, content: bytes) -> Dict[str, Any]:
    """
    Recebe bytes de imagem, tenta localizar e extrair protocolo.
    Retorna dict pronto para ser salvo/serializado.
    """
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "arquivo": filename,
            "protocolo_ocr": None,
            "protocolo_17_digitos": None,
            "data_extracao": now_belem_str(),
            "ok": False,
        }

    # 1) Tenta ROI ao redor de "PROTOC..."
    textos = ""
    roi = find_protocolo_roi(img)
    if roi:
        x1, y1, x2, y2 = roi
        recorte = img[y1:y2, x1:x2]
        textos = tentar_ocr_em_varias_formas(recorte)

    # 2) Se falhar, tenta na imagem inteira
    if not textos.strip():
        textos = tentar_ocr_em_varias_formas(img)

    protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(textos)

    # 3) Tentativa extra com OCR "full" (sem digits_only) se ainda não obteve
    if not protocolo_17:
        texto_extra = ocr_text(img, tesseract_config(digits_only=False))
        if texto_extra:
            protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(texto_extra)

    ok = bool(protocolo_17)
    return {
        "arquivo": filename,
        "protocolo_ocr": protocolo_fmt,
        "protocolo_17_digitos": protocolo_17,
        "data_extracao": now_belem_str(),
        "ok": ok,
    }


# -------------------------
# Caminho do Excel (usado pelo main_web.py)
# -------------------------
EXCEL_PATH = Path(__file__).parent / "data" / "protocolos_extraidos.xlsx"
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Compat: funções de consolidação legadas (opcional)
# Observação: o main_web.py novo NÃO usa estas funções;
# elas permanecem aqui apenas para não quebrar chamadas antigas.
# -------------------------
def _only_digits_17(x: Any) -> Optional[str]:
    s = "" if x is None else str(x)
    d = "".join(ch for ch in s if ch.isdigit())
    return d if len(d) == 17 else None


def salvar_excel_texto(df_total: pd.DataFrame) -> None:
    """Grava DataFrame como Excel forçando 'protocolo_17_digitos' a TEXTO."""
    if "protocolo_17_digitos" not in df_total.columns:
        df_total["protocolo_17_digitos"] = ""
    else:
        df_total["protocolo_17_digitos"] = df_total["protocolo_17_digitos"].fillna("").astype(str)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="w") as writer:
        df_total.to_excel(writer, index=False, sheet_name="Sheet1")
        ws = writer.book["Sheet1"]
        try:
            col_idx = list(df_total.columns).index("protocolo_17_digitos") + 1
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=col_idx).number_format = "@"
        except Exception:
            # se por algum motivo a coluna não estiver presente
            pass


def _carregar_set_existentes() -> set[str]:
    existentes: set[str] = set()
    if EXCEL_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_PATH, dtype=str, engine="openpyxl")
            if "protocolo_17_digitos" in df.columns:
                for v in df["protocolo_17_digitos"].astype(str).fillna(""):
                    v17 = _only_digits_17(v)
                    if v17:
                        existentes.add(v17)
        except Exception:
            pass
    return existentes


def consolidar_resultados(novos: List[dict]) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    """
    [CORRIGIDO] Consolida no Excel, removendo duplicatas de forma mais robusta.
    Retorna: (df_total, duplicados_excel, inseridos, duplicados_lote)
    """
    colunas = ["arquivo", "protocolo_ocr", "protocolo_17_digitos", "data_extracao"]
    df_novos_full = pd.DataFrame(novos)

    # 1. Carrega protocolos existentes e os novos para conjuntos (sets)
    set_existentes = _carregar_set_existentes()
    
    # Prepara os dados novos, filtrando e formatando
    inserir_rows: List[Dict] = []
    vistos_lote: set[str] = set()
    duplicados_lote: List[Dict] = []
    
    for _, row in df_novos_full.iterrows():
        proto_raw = row.get("protocolo_17_digitos", "")
        proto_17 = _only_digits_17(proto_raw)
        
        # Ignora protocolos inválidos ou nulos
        if not proto_17:
            continue
            
        # Verifica duplicatas dentro do lote
        if proto_17 in vistos_lote:
            duplicados_lote.append({
                "arquivo": row.get("arquivo", ""),
                "protocolo_17_digitos": proto_17
            })
            continue

        vistos_lote.add(proto_17)
        
        # Adiciona à lista de inserção se não for duplicado no lote
        item = {
            "arquivo": row.get("arquivo", ""),
            "protocolo_ocr": row.get("protocolo_ocr", ""),
            "protocolo_17_digitos": proto_17,
            "data_extracao": row.get("data_extracao", now_belem_str())
        }
        inserir_rows.append(item)

    # 2. Compara com os protocolos existentes no Excel
    inserir_unicos: List[Dict] = []
    duplicados_excel: List[Dict] = []
    
    for item in inserir_rows:
        p17 = item["protocolo_17_digitos"]
        if p17 in set_existentes:
            duplicados_excel.append({
                "arquivo": item.get("arquivo", ""),
                "protocolo_17_digitos": p17
            })
        else:
            inserir_unicos.append(item)
            set_existentes.add(p17)

    # 3. Concatena e salva
    df_existente = pd.read_excel(EXCEL_PATH, dtype=str, engine="openpyxl") if EXCEL_PATH.exists() else pd.DataFrame(columns=colunas)
    df_inserir = pd.DataFrame(inserir_unicos, columns=colunas)
    df_total = pd.concat([df_existente, df_inserir], ignore_index=True)

    salvar_excel_texto(df_total)
    
    # 4. Retorna os resultados
    return df_total, duplicados_excel, inserir_unicos, duplicados_lote