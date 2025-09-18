from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import uvicorn

from .ocr_core import processar_imagem_bytes, consolidar_resultados, EXCEL_PATH

app = FastAPI(title="Protocolo OCR Web", version="1.0")

# Libere o front-end estático local ou seu domínio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # troque por seu domínio em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>API Protocolo OCR</h1><p>Use POST /extract</p>"

@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    resultados = []
    for f in files:
        content = await f.read()
        r = processar_imagem_bytes(f.filename, content)
        resultados.append(r)
    df_total = consolidar_resultados(resultados)
    total_ok = int(df_total['protocolo_17_digitos'].replace({"": None}).dropna().shape[0])
    return JSONResponse({
        "processados": len(resultados),
        "total_protocolos_no_excel": total_ok,
        "excel_path": "/download/excel",
        "resultados_lote": resultados
    })

@app.get("/download/excel")
def download_excel():
    if not EXCEL_PATH.exists():
        return JSONResponse({"erro":"Excel ainda não existe"}, status_code=404)
    return FileResponse(EXCEL_PATH, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=EXCEL_PATH.name)

if __name__ == "__main__":
    uvicorn.run("app.main_web:app", host="0.0.0.0", port=8000, reload=True)
