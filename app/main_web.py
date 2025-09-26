# main_web.py - ATUALIZADO PARA SETOR/PROFISSIONAL/NOME

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import uvicorn
import time

from .ocr_core import (
    processar_imagem_bytes, 
    consolidar_resultados, 
    ler_protocolos_existentes,
    EXCEL_PATH
)

app = FastAPI(title="Protocolo OCR Web", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>API Protocolo OCR Rodando</h1><p>Use POST /extract</p>"

@app.post("/extract")
async def extract(
    files: List[UploadFile] = File(...),
    setor: str = Form(...),
    profissional: str = Form(...),
    nome_completo: str = Form(...)
):
    t0 = time.time()

    # Normalização simples
    setor = (setor or "").strip()
    profissional = (profissional or "").strip()
    nome_completo = (nome_completo or "").strip()

    # Leitura do histórico (com robustez)
    protocolos_existentes, df_existente = ler_protocolos_existentes()

    resultados = []
    protocolos_do_lote_set = set()

    encontrados_lote = 0
    novos_adicionados = 0
    duplicados_ignorar = []
    resultados_para_salvar = []

    for i, f in enumerate(files, start=1):
        content = await f.read()
        r = processar_imagem_bytes(f.filename, content)

        # Anexa os novos metadados de identificação do usuário/uso
        r["setor"] = setor
        r["profissional"] = profissional
        r["nome_completo"] = nome_completo

        protocolo_17 = r.get("protocolo_17_digitos")
        is_duplicate = False

        if protocolo_17:
            encontrados_lote += 1
            if protocolo_17 in protocolos_existentes or protocolo_17 in protocolos_do_lote_set:
                is_duplicate = True
                duplicados_ignorar.append(protocolo_17)
            else:
                protocolos_do_lote_set.add(protocolo_17)
                novos_adicionados += 1
                resultados_para_salvar.append(r)

        r["indice_processamento"] = i
        r["ok"] = bool(protocolo_17)
        r["duplicado"] = is_duplicate
        resultados.append(r)

    # Consolida apenas os não-duplicados
    df_total = consolidar_resultados(resultados_para_salvar, df_existente)

    total_protocolos_validos_excel = int(df_total['protocolo_17_digitos'].str.len().eq(17).sum())
    duplicados_unicos_ignorar = sorted(list(set(duplicados_ignorar)))

    resumo = {
        "total_docs": len(resultados),
        "processados": len(resultados),
        "encontrados_lote": encontrados_lote,
        "novos_adicionados": novos_adicionados,
        "duplicados_ignorar": len(duplicados_ignorar),
        "duplicados_lista": duplicados_unicos_ignorar,
        "tempo_total_s": round(time.time() - t0, 3)
    }

    return JSONResponse({
        "resumo": resumo,
        "processados": len(resultados),
        "total_protocolos_no_excel": total_protocolos_validos_excel,
        "excel_path": "/download/excel",
        "resultados_lote": resultados
    })

@app.get("/download/excel")
def download_excel():
    if not EXCEL_PATH.exists():
        return JSONResponse({"erro": "Excel ainda não existe"}, status_code=404)
    return FileResponse(
        EXCEL_PATH,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=EXCEL_PATH.name
    )

if __name__ == "__main__":
    uvicorn.run("main_web:app", host="0.0.0.0", port=8000, reload=True)
