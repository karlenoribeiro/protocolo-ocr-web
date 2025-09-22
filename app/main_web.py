# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openpyxl.utils import get_column_letter

from .ocr_core import processar_imagem_bytes, agora_belem_str


# -----------------------------------------------------------------------------
# Configuração básica
# -----------------------------------------------------------------------------
app = FastAPI(title="Protocolo OCR Web", version="1.0.0")

# Habilita CORS conforme necessidade
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pasta/arquivo para planilha consolidada
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = DATA_DIR / "protocolos_extraidos.xlsx"

# Colunas padrão do dataset
COLS = [
    "protocolo_17_digitos",
    "arquivo",
    "detectado_em",
]


# -----------------------------------------------------------------------------
# Utilitários de persistência (Excel)
# -----------------------------------------------------------------------------
def _carrega_excel() -> pd.DataFrame:
    if EXCEL_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_PATH, dtype=str).fillna("")
            # garante colunas previstas
            for c in COLS:
                if c not in df.columns:
                    df[c] = ""
            return df[COLS]
        except Exception:
            # se der algum erro, começa limpo
            pass
    return pd.DataFrame(columns=COLS)


def _salva_excel_texto(path: Path, df: pd.DataFrame) -> None:
    """
    Salva o DataFrame no Excel e força a coluna protocolo como 'texto' (nunca notação científica).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name="dados")
        ws = writer.book["dados"]

        # Tenta encontrar a coluna e aplicar number_format="@"
        try:
            col_idx = df.columns.get_loc("protocolo_17_digitos") + 1  # 1-based
            col_letter = get_column_letter(col_idx)
            for row in ws[f"{col_letter}:{col_letter}"]:
                for cell in row:
                    cell.number_format = "@"
        except Exception:
            # se não existir, segue
            pass


# -----------------------------------------------------------------------------
# Rotas
# -----------------------------------------------------------------------------
@app.get("/ping")
def ping() -> Dict[str, Any]:
    return {
        "ok": True,
        "app": "Protocolo OCR Web",
        "versao": app.version,
        "agora_belem": agora_belem_str(),
    }


@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Recebe 1..N imagens e tenta extrair o protocolo (17 dígitos).
    – Grava/atualiza planilha consolidada (evita duplicados).
    – Retorna um resumo da operação e os resultados individuais.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Envie ao menos 1 arquivo de imagem.")

    resultados: List[Dict[str, Any]] = []

    for f in files:
        content = await f.read()
        r = processar_imagem_bytes(content, f.filename or "")
        resultados.append(r)

    # Constrói DF apenas dos itens OK com protocolo detectado
    df_novos = pd.DataFrame(
        [
            {
                "protocolo_17_digitos": r["protocolo_17_digitos"],
                "arquivo": r.get("arquivo", ""),
                "detectado_em": agora_belem_str(),
            }
            for r in resultados
            if r.get("ok") and r.get("protocolo_17_digitos")
        ],
        columns=COLS,
    )

    # Carrega existentes e evita duplicar
    df_exist = _carrega_excel()

    duplicados = []
    inseridos = []
    if not df_novos.empty:
        # protocolos já existentes
        protocolos_exist = set(df_exist["protocolo_17_digitos"].astype(str).tolist())
        for _, row in df_novos.iterrows():
            p = str(row["protocolo_17_digitos"])
            if p in protocolos_exist:
                duplicados.append(p)
            else:
                inseridos.append(p)

        # concatena somente os inéditos
        df_final = pd.concat(
            [df_exist, df_novos[~df_novos["protocolo_17_digitos"].isin(duplicados)]],
            ignore_index=True,
        )
        _salva_excel_texto(EXCEL_PATH, df_final)
    else:
        df_final = df_exist  # nada novo

    resumo = {
        "total_recebidos": len(files),
        "ok": sum(1 for r in resultados if r.get("ok")),
        "com_protocolo": sum(1 for r in resultados if r.get("protocolo_17_digitos")),
        "inseridos": inseridos,
        "duplicados": duplicados,
        "planilha_atual": str(EXCEL_PATH.resolve()),
    }

    return JSONResponse(
        {
            "resumo": resumo,
            "resultados_lote": resultados,
        }
    )


@app.get("/download/excel")
def download_excel() -> StreamingResponse:
    """
    Baixa a planilha consolidada; se ainda não existir, entrega vazia.
    """
    df = _carrega_excel()
    buf = io.BytesIO()
    _salva_excel_texto(EXCEL_PATH, df)  # mantém arquivo no disco atualizado
    with pd.ExcelWriter(buf, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name="dados")
    buf.seek(0)

    filename = "protocolos_extraidos.xlsx"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers
    )
