# =======================
# EXTRAÇÃO DE PROTOCOLOS (17 DÍGITOS) DE IMAGENS
# Mantém sua lógica; acrescenta deduplicação INTRA-LOTE e separa duplicados_lote vs. duplicados_excel
# =======================

import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
from unidecode import unidecode
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from openpyxl import load_workbook

LOCAL_TZ = ZoneInfo("America/Belem")
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jfif"}

def now_belem_str():
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

def _langs():
    try:
        langs = set(pytesseract.get_languages(config=""))
        return "por+eng" if "por" in langs else "eng"
    except Exception:
        return "por+eng"

OCR_LANGS = _langs()

def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[1, 0])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)

def deskew_osd(img):
    try:
        osd = pytesseract.image_to_osd(img)
        angle_m = re.search(r"Rotate: (\d+)", osd)
        if angle_m:
            angle = float(angle_m.group(1)) % 360
            if angle > 1:
                return rotate_bound(img, angle)
    except Exception:
        pass
    return img

def enhance_for_ocr(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def tesseract_config(digits_only=False):
    base = f"--oem 3 --psm 6 -l {OCR_LANGS}"
    if digits_only:
        base += " -c tessedit_char_whitelist=0123456789-/–"
    return base

def ocr_text(img, config):
    return pytesseract.image_to_string(img, config=config)

def ocr_data(img, config):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME, config=config)

def find_protocolo_roi(img) -> Optional[Tuple[int,int,int,int]]:
    df = ocr_data(img, tesseract_config(digits_only=False))
    try:
        df = df.dropna(subset=['text'])
    except Exception:
        return None
    if df is None or len(df)==0:
        return None
    df['norm'] = df['text'].astype(str).apply(lambda s: unidecode(s).upper())
    mask = df['norm'].str.contains(r'^PROTOC', regex=True)
    cand = df[mask]
    if cand.empty:
        return None
    cand = cand.sort_values(['top','left']).iloc[0]
    x, y, w, h = int(cand['left']), int(cand['top']), int(cand['width']), int(cand['height'])
    H, W = img.shape[:2]
    pad_y = int(h*2.5)
    pad_x_left = int(w*0.5)
    x1 = max(0, x - pad_x_left)
    y1 = max(0, y - pad_y//2)
    x2 = W
    y2 = min(H, y + pad_y)
    return (x1, y1, x2, y2)

PROTO_REGEXES = [
    re.compile(r'PROTOCOLO[:\s-]*([0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2})', re.IGNORECASE),
    re.compile(r'([0-9]{5}\s*[-–]?\s*[0-9]{6}\s*/\s*[0-9]{4}\s*[-–]?\s*[0-9]{2})')
]

def limpar_para_17_digitos(seq: str) -> Optional[str]:
    if not seq:
        return None
    digits = re.sub(r'\D', '', seq)
    return digits if len(digits) == 17 else None

def extrair_protocolo_de_texto(texto: str):
    if not texto:
        return (None, None)
    for rgx in PROTO_REGEXES:
        m = rgx.search(texto)
        if m:
            bruto = m.group(1)
            apenas_numeros = limpar_para_17_digitos(bruto)
            if apenas_numeros:
                fmt = f"{apenas_numeros[0:5]}-{apenas_numeros[5:11]}/{apenas_numeros[11:15]}-{apenas_numeros[15:17]}"
                return (fmt, apenas_numeros)
            return (bruto, None)
    return (None, None)

def tentar_ocr_em_varias_formas(img_color):
    gray1 = to_gray(deskew_osd(img_color))
    th = enhance_for_ocr(gray1)
    inv = cv2.bitwise_not(th)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    cfg = tesseract_config(digits_only=True)
    textos = []
    for candidate in [gray1, th, inv, morph]:
        try:
            textos.append(ocr_text(candidate, cfg))
        except Exception:
            pass
    return "\n".join([t for t in textos if t])

def processar_imagem_bytes(filename: str, content: bytes):
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "arquivo": filename,
            "protocolo_ocr": None,
            "protocolo_17_digitos": None,
            "data_extracao": now_belem_str(),
            "ok": False
        }

    roi = find_protocolo_roi(img)
    textos = ""
    if roi:
        x1,y1,x2,y2 = roi
        recorte = img[y1:y2, x1:x2]
        textos = tentar_ocr_em_varias_formas(recorte)

    if not textos.strip():
        textos = tentar_ocr_em_varias_formas(img)

    protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(textos)

    if not protocolo_17:
        try:
            texto_extra = ocr_text(img, tesseract_config(digits_only=False))
            protocolo_fmt, protocolo_17 = extrair_protocolo_de_texto(texto_extra)
        except Exception:
            pass

    ok = bool(protocolo_17)
    return {
        "arquivo": filename,
        "protocolo_ocr": protocolo_fmt,
        "protocolo_17_digitos": protocolo_17,
        "data_extracao": now_belem_str(),
        "ok": ok
    }

EXCEL_PATH = Path(__file__).parent / "data" / "protocolos_extraidos.xlsx"
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- carrega protocolos existentes como set ---
def _carregar_set_existentes() -> set:
    """Carrega 'protocolo_17_digitos' já existentes no Excel como set de strings."""
    existentes = set()
    if EXCEL_PATH.exists():
        try:
            df_exist = pd.read_excel(EXCEL_PATH, dtype={"protocolo_17_digitos": str})
            if "protocolo_17_digitos" in df_exist.columns:
                for v in df_exist["protocolo_17_digitos"].astype(str).fillna(""):
                    v = "".join(ch for ch in v if ch.isdigit())
                    if len(v) == 17:
                        existentes.add(v)
        except Exception:
            pass  # se der erro, considera vazio
    return existentes

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
        if col_idx is not None:
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.number_format = "@"

def consolidar_resultados(novos: List[dict]) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    """
    Consolida no Excel com três estágios:
      (A) Deduplica DENTRO do lote -> 'duplicados_lote'
      (B) Compara com Excel (set_existentes) -> 'duplicados_excel'
      (C) Salva df_total com drop_duplicates final por segurança

    Retorna: (df_total, duplicados_excel, inseridos, duplicados_lote)
    """
    colunas = ["arquivo","protocolo_ocr","protocolo_17_digitos","data_extracao"]

    df_novos_full = pd.DataFrame(novos)
    df_novos = df_novos_full.reindex(columns=colunas)

    # (1) Carrega o Excel existente (se houver)
    if EXCEL_PATH.exists():
        try:
            df_existente = pd.read_excel(EXCEL_PATH, dtype={"protocolo_17_digitos": str})
        except Exception:
            df_existente = pd.DataFrame(columns=colunas)
    else:
        df_existente = pd.DataFrame(columns=colunas)

    # (2) Monta set com protocolos já no Excel
    set_existentes = _carregar_set_existentes()

    # Helper para normalizar para 17 dígitos
    def only_digits_17(x: str) -> Optional[str]:
        if not isinstance(x, str):
            return None
        d = "".join(ch for ch in x if ch.isdigit())
        return d if len(d) == 17 else None

    # (A) DEDUPLICAÇÃO INTRA-LOTE
    vistos_no_lote: set[str] = set()
    duplicados_lote: List[Dict] = []
    candidatos_para_excel: List[Dict] = []

    for _, row in df_novos.iterrows():
        proto_raw = row.get("protocolo_17_digitos", "")
        proto_17 = only_digits_17(str(proto_raw)) if proto_raw else None
        if not proto_17:
            # sem 17 dígitos válidos -> ignora silenciosamente
            continue

        if proto_17 in vistos_no_lote:
            # repetido no mesmo upload
            duplicados_lote.append({
                "arquivo": row.get("arquivo", ""),
                "protocolo_17_digitos": proto_17
            })
            continue

        # primeiro encontro no lote -> segue adiante
        vistos_no_lote.add(proto_17)
        candidatos_para_excel.append({
            "arquivo": row.get("arquivo", ""),
            "protocolo_ocr": row.get("protocolo_ocr", ""),
            "protocolo_17_digitos": proto_17,
            "data_extracao": row.get("data_extracao", "")
        })

    # (B) COMPARAÇÃO COM O EXCEL
    duplicados_excel: List[Dict] = []
    inserir_rows: List[Dict] = []

    for item in candidatos_para_excel:
        proto_17 = item["protocolo_17_digitos"]
        if proto_17 in set_existentes:
            duplicados_excel.append({
                "arquivo": item.get("arquivo", ""),
                "protocolo_17_digitos": proto_17
            })
        else:
            inserir_rows.append(item)
            set_existentes.add(proto_17)

    # (C) CONCATENA E SANEIA
    df_inserir = pd.DataFrame(inserir_rows, columns=colunas)
    df_total = pd.concat([df_existente, df_inserir], ignore_index=True)

    if "protocolo_17_digitos" in df_total.columns:
        df_total["protocolo_17_digitos"] = df_total["protocolo_17_digitos"].astype(str)
        df_total = df_total.drop_duplicates(subset=["protocolo_17_digitos"], keep="first")

    salvar_excel_texto(df_total)
    return df_total, duplicados_excel, inserir_rows, duplicados_lote
