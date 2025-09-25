# main_web.py - CÓDIGO CORRIGIDO

# ============================================================
# API PARA EXTRAÇÃO DE PROTOCOLOS (OCR) — com resumo do lote
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import uvicorn
import time

# Importa as novas funções/tipos do módulo interno de OCR:
from .ocr_core import (
    processar_imagem_bytes, 
    consolidar_resultados, 
    ler_protocolos_existentes, # NOVA FUNÇÃO
    EXCEL_PATH
)

app = FastAPI(title="Protocolo OCR Web", version="1.1")

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

# Rota POST "/extract" que processa os arquivos enviados
@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    t0 = time.time()
    
    # 1. Pré-leitura: LÊ PROTOCOLOS EXISTENTES (set para checagem rápida)
    protocolos_existentes, df_existente = ler_protocolos_existentes()
    
    resultados = []
    protocolos_do_lote_set = set() # Para checar duplicatas dentro do lote atual (memória)
    
    # Contadores
    encontrados_lote = 0 # Protocolos válidos encontrados neste lote (antes da checagem)
    novos_adicionados = 0 # Protocolos novos (não duplicados no Excel ou no lote)
    duplicados_ignorar = [] # Lista de protocolos duplicados a ignorar
    
    # Lista dos resultados válidos e não duplicados que serão salvos
    resultados_para_salvar = []

    # Loop para processar cada arquivo enviado
    for i, f in enumerate(files, start=1):
        content = await f.read()
        r = processar_imagem_bytes(f.filename, content)
        
        protocolo_17 = r.get("protocolo_17_digitos")
        
        # 2. Lógica de Checagem de Duplicação
        is_duplicate = False
        
        if protocolo_17:
            encontrados_lote += 1
            # Checa se é duplicado no histórico (Excel) OU no lote atual (memória)
            if protocolo_17 in protocolos_existentes or protocolo_17 in protocolos_do_lote_set:
                is_duplicate = True
                duplicados_ignorar.append(protocolo_17)
            else:
                # É um protocolo novo. Adiciona ao set do lote para checar futuras duplicações neste mesmo POST
                protocolos_do_lote_set.add(protocolo_17)
                novos_adicionados += 1
                # Adiciona à lista de salvamento (apenas os que não são duplicados)
                resultados_para_salvar.append(r)

        r["indice_processamento"] = i
        r["ok"] = bool(protocolo_17) # Marca se encontrou o protocolo
        r["duplicado"] = is_duplicate # Adiciona info de duplicação para o resumo no frontend
        resultados.append(r)

    # 3. Consolidação no Excel
    # Passa APENAS os resultados válidos E NÃO DUPLICADOS e o DataFrame existente
    df_total = consolidar_resultados(resultados_para_salvar, df_existente)
    
    # 4. Finaliza Resumo e Resposta
    # Contabiliza o total de protocolos válidos no Excel (len==17)
    total_protocolos_validos_excel = int(df_total['protocolo_17_digitos'].str.len().eq(17).sum())
    
    # Remove duplicatas da lista de ignorados para o resumo
    duplicados_unicos_ignorar = sorted(list(set(duplicados_ignorar)))

    resumo = {
        "total_docs": len(resultados),
        "processados": len(resultados),
        "encontrados_lote": encontrados_lote, # Total de protocolos válidos encontrados no lote
        "novos_adicionados": novos_adicionados, # Total de protocolos NOVOS adicionados (não duplicados)
        "duplicados_ignorar": len(duplicados_ignorar), # Total de duplicados IGNORADOS (do lote)
        "duplicados_lista": duplicados_unicos_ignorar, # Lista dos protocolos duplicados ignorados
        "tempo_total_s": round(time.time() - t0, 3)
    }

    # Retorna a resposta em JSON contendo o resumo e os resultados detalhados
    return JSONResponse({
        "resumo": resumo,
        "processados": len(resultados),
        "total_protocolos_no_excel": total_protocolos_validos_excel,
        "excel_path": "/download/excel",
        "resultados_lote": resultados
    })

# Rota GET para download do arquivo Excel consolidado (Mantida)
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