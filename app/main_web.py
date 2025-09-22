# ============================================================
# API PARA EXTRAÇÃO DE PROTOCOLOS (OCR) — com resumo do lote
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import time

from .ocr_core import processar_imagem_bytes, consolidar_resultados, EXCEL_PATH

app = FastAPI(title="Protocolo OCR Web", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste em produção para domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>API Protocolo OCR</h1><p>Use POST /extract</p>"

@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    t0 = time.time()
    resultados = []
    encontrados = 0

    # Processa cada arquivo enviado
    for i, f in enumerate(files, start=1):
        content = await f.read()
        r = processar_imagem_bytes(f.filename, content)
        r["indice_processamento"] = i  # útil para auditoria no front
        resultados.append(r)
        if r.get("protocolo_17_digitos"):
            encontrados += 1

    # Consolidação (dedupe no lote + dedupe contra Excel)
    df_total, duplicados_excel, inseridos, duplicados_lote = consolidar_resultados(resultados)

    # Total de protocolos válidos (17 dígitos) presentes no Excel após a consolidação
    total_ok_excel = 0
    if "protocolo_17_digitos" in df_total.columns:
        total_ok_excel = int(
            df_total["protocolo_17_digitos"]
            .astype(str)
            .apply(lambda s: s.isdigit() and len(s) == 17)
            .sum()
        )

    # Resumo das operações do lote
    resumo = {
        "total_docs": len(resultados),
        "processados": len(resultados),
        "encontrados": encontrados,
        "nao_encontrados": len(resultados) - encontrados,
        "duplicados_no_lote": len(duplicados_lote),
        "duplicados_no_excel": len(duplicados_excel),
        "inseridos_no_excel": len(inseridos),
        "tempo_total_s": round(time.time() - t0, 3),
    }

    return JSONResponse({
        "resumo": resumo,
        "processados": len(resultados),
        "total_protocolos_no_excel": total_ok_excel,
        "excel_path": "/download/excel",
        "resultados_lote": resultados,
        "duplicados_lote": duplicados_lote,
        "duplicados": duplicados_excel,   # mantém a chave antiga para compatibilidade
        "inseridos": inseridos
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
    uvicorn.run("app.main_web:app", host="0.0.0.0", port=8000, reload=True)
