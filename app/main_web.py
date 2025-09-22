# -*- coding: utf-8 -*-
"""
API de OCR de protocolo com:
- Deduplicação intra-lote e contra Excel
- Relatório do processamento
- Endpoint para baixar a planilha
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from .ocr_core import processar_imagem_bytes, agora_belem_str

APP_VERSION = "1.3.0"

# Diretórios / arquivos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = DATA_DIR / "protocolos_extraidos.xlsx"

# Colunas-padrão da planilha
COLS = ["arquivo", "protocolo_ocr", "protocolo_17_digitos", "data_extracao"]

# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------

def _normaliza_proto_17(v: Any) -> str:
    """
    - Garante string
    - Mantém somente dígitos
    - Retorna '' se não tiver exatamente 17 dígitos
    """
    if v is None:
        return ""
    s = str(v)
    dig = "".join(ch for ch in s if ch.isdigit())
    return dig if len(dig) == 17 else ""


def _carrega_excel(path: Path) -> pd.DataFrame:
    """Carrega o Excel se existir, garantindo colunas e dtype=str."""
    if path.exists():
        try:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
        except Exception:
            # Arquivo pode estar corrompido; tenta recuperar mínimo
            df = pd.DataFrame(columns=COLS)
    else:
        df = pd.DataFrame(columns=COLS)
    # Garante colunas na ordem
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS].astype(str)


def _salva_excel_texto(path: Path, df: pd.DataFrame) -> None:
    """
    Salva DataFrame no Excel garantindo a coluna 'protocolo_17_digitos' como TEXTO.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Escreve com openpyxl e depois ajusta o number_format da coluna
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name="dados")
        ws = writer.book["dados"]
        # Descobre índice da coluna 'protocolo_17_digitos'
        try:
            col_idx = df.columns.get_loc("protocolo_17_digitos") + 1  # 1-based
            col_letter = get_column_letter(col_idx)
            for cell in ws[f"{col_letter}:{col_letter}"]:
                cell.number_format = "@"
        except Exception:
            pass


def salvar_excel_sem_duplicar(path: Path, df_lote: pd.DataFrame) -> Dict[str, Any]:
    """
    Consolida os resultados do lote no Excel, impedindo duplicados.

    - Remove duplicados intra-lote (mesmo protocolo repetido nas imagens enviadas agora)
    - Não reinsere protocolos que já existem no Excel
    - Retorna estatísticas e listas para exibição na UI
    """
    # Normaliza a coluna de protocolo do lote
    df_lote = df_lote.copy()
    if not len(df_lote):
        # Nada a salvar
        return {
            "duplicados_no_lote": 0,
            "ja_salvos": 0,
            "inseridos": 0,
            "lista_duplicados_lote": [],
            "lista_ja_salvos": [],
            "lista_inseridos": [],
        }

    df_lote["protocolo_17_digitos"] = df_lote["protocolo_17_digitos"].map(_normaliza_proto_17)
    df_lote["data_extracao"] = df_lote["data_extracao"].fillna(agora_belem_str())

    # Filtra válidos (17 dígitos)
    df_validos = df_lote[df_lote["protocolo_17_digitos"].str.len() == 17].copy()

    # Deduplicação intra-lote
    antes = len(df_validos)
    # Mantém a PRIMEIRA ocorrência; duplicados do mesmo lote são reportados
    df_validos = df_validos.drop_duplicates(subset=["protocolo_17_digitos"], keep="first")
    dup_intra_count = antes - len(df_validos)
    # Quais foram duplicados no lote:
    contagens_lote = (
        df_lote[df_lote["protocolo_17_digitos"].str.len() == 17]["protocolo_17_digitos"]
        .value_counts()
    )
    lista_duplicados_lote = sorted([p for p, n in contagens_lote.items() if n > 1])

    # Carrega o Excel existente
    df_excel = _carrega_excel(path)

    # Conjunto já salvo
    ja_no_excel = set(df_excel["protocolo_17_digitos"].map(_normaliza_proto_17))

    # Novos a inserir = válidos do lote que ainda não estão no Excel
    mask_novos = ~df_validos["protocolo_17_digitos"].isin(ja_no_excel)
    df_novos = df_validos[mask_novos].copy()
    df_repetidos_excel = df_validos[~mask_novos].copy()

    # Listas para a UI
    lista_inseridos = sorted(df_novos["protocolo_17_digitos"].unique().tolist())
    lista_ja_salvos = sorted(df_repetidos_excel["protocolo_17_digitos"].unique().tolist())

    # Concatena e salva
    if len(df_novos):
        df_out = pd.concat([df_excel, df_novos[COLS]], ignore_index=True)
    else:
        df_out = df_excel

    _salva_excel_texto(path, df_out)

    return {
        "duplicados_no_lote": dup_intra_count,
        "ja_salvos": len(lista_ja_salvos),
        "inseridos": len(lista_inseridos),
        "lista_duplicados_lote": lista_duplicados_lote,
        "lista_ja_salvos": lista_ja_salvos,
        "lista_inseridos": lista_inseridos,
    }

# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------

app = FastAPI(title="Protocolo OCR API", version=APP_VERSION)

# CORS (em produção, restrinja para seu domínio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # troque por ["https://seu-dominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"ok": True, "version": APP_VERSION, "now": agora_belem_str()}


@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Recebe imagens, aplica OCR, extrai protocolo, consolida Excel sem duplicar e
    retorna:
      - resumo (totais do lote)
      - total_protocolos_no_excel (acumulado)
      - listas: duplicados_lote, duplicados (no Excel), inseridos
      - resultados_lote (por arquivo)
    """
    t0 = time.time()
    resultados: List[Dict[str, Any]] = []
    encontrados = 0

    for i, f in enumerate(files, start=1):
        content = await f.read()
        r = processar_imagem_bytes(f.filename, content)
        r["indice_processamento"] = i
        if not r.get("data_extracao"):
            r["data_extracao"] = agora_belem_str()
        # normaliza p17
        p17 = _normaliza_proto_17(r.get("protocolo_17_digitos"))
        r["protocolo_17_digitos"] = p17
        if p17:
            encontrados += 1
        resultados.append(r)

    # DataFrame do lote (com as colunas padronizadas)
    df_lote = pd.DataFrame([{
        "arquivo": r.get("arquivo", r.get("filename", "")) or "",
        "protocolo_ocr": r.get("protocolo_ocr", "") or "",
        "protocolo_17_digitos": r.get("protocolo_17_digitos", "") or "",
        "data_extracao": r.get("data_extracao", agora_belem_str()),
    } for r in resultados], columns=COLS)

    stats = salvar_excel_sem_duplicar(EXCEL_PATH, df_lote)

    # Total acumulado válido (17 dígitos) no Excel
    total_ok_excel = 0
    try:
        df_total = pd.read_excel(EXCEL_PATH, dtype=str, engine="openpyxl")
        if "protocolo_17_digitos" in df_total.columns:
            total_ok_excel = int(
                df_total["protocolo_17_digitos"]
                .astype(str)
                .apply(lambda s: s.isdigit() and len(s) == 17)
                .sum()
            )
    except Exception:
        pass

    resumo = {
        "total_docs": len(resultados),
        "processados": len(resultados),
        "encontrados": encontrados,
        "nao_encontrados": len(resultados) - encontrados,
        "duplicados_no_lote": stats["duplicados_no_lote"],
        "duplicados_no_excel": stats["ja_salvos"],
        "inseridos_no_excel": stats["inseridos"],
        "tempo_total_s": round(time.time() - t0, 3),
    }

    return JSONResponse({
        "resumo": resumo,
        "processados": len(resultados),
        "total_protocolos_no_excel": total_ok_excel,
        "excel_path": "/download/excel",
        "resultados_lote": resultados,
        "duplicados_lote": stats["lista_duplicados_lote"],
        "duplicados": stats["lista_ja_salvos"],
        "inseridos": stats["lista_inseridos"],
    })


@app.get("/download/excel")
def download_excel():
    """
    Baixa a planilha consolidada. Se ainda não existir, devolve planilha vazia.
    """
    if not EXCEL_PATH.exists():
        df = pd.DataFrame(columns=COLS)
        _salva_excel_texto(EXCEL_PATH, df)
    return FileResponse(EXCEL_PATH, filename=EXCEL_PATH.name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
