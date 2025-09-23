# ============================================================
# API PARA EXTRAÇÃO DE PROTOCOLOS (OCR) — com resumo do lote
# ============================================================

# Importa classes e funções do framework FastAPI para criar a API
from fastapi import FastAPI, UploadFile, File
# Importa classes para respostas HTTP: arquivos, JSON e HTML
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
# Middleware para permitir comunicação entre diferentes origens (CORS)
from fastapi.middleware.cors import CORSMiddleware
# Manipulação de caminhos de arquivos de forma segura e cross-platform
from pathlib import Path
# Tipagem para listas e outros tipos avançados
from typing import List
# Servidor ASGI utilizado para rodar a aplicação FastAPI
import uvicorn
# Biblioteca para medir tempo de execução
import time

# Importa funções do módulo interno de OCR:
# - processar_imagem_bytes: processa imagens e extrai protocolos
# - consolidar_resultados: salva e consolida dados em planilha Excel
# - EXCEL_PATH: caminho do arquivo Excel utilizado
from .ocr_core import processar_imagem_bytes, consolidar_resultados, EXCEL_PATH

# Cria a instância principal da aplicação FastAPI
# Define título e versão que aparecerão na documentação automática
app = FastAPI(title="Protocolo OCR Web", version="1.1")

# Adiciona configuração de CORS para permitir acesso à API via frontend ou outros domínios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" permite qualquer origem (ajustar em produção por segurança)
    allow_credentials=True,  # Permite envio de cookies/autenticação
    allow_methods=["*"],  # Permite qualquer método HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite qualquer cabeçalho customizado
)

# Rota GET principal ("/") que retorna um HTML simples com instruções
@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>API Protocolo OCR</h1><p>Use POST /extract</p>"

# Rota POST "/extract" que processa os arquivos enviados
@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    # Marca o tempo inicial para medir o tempo total de processamento
    t0 = time.time()
    # Lista para armazenar os resultados de cada imagem processada
    resultados = []
    # Contador para saber quantos arquivos tiveram protocolo encontrado
    encontrados = 0

    # Loop para processar cada arquivo enviado
    # enumerate(files, start=1) gera índice a partir de 1 para cada arquivo
    for i, f in enumerate(files, start=1):
        # Lê o conteúdo binário do arquivo enviado
        content = await f.read()
        # Chama a função OCR para processar a imagem
        r = processar_imagem_bytes(f.filename, content)
        # Adiciona índice de processamento ao resultado para controle
        r["indice_processamento"] = i
        # Adiciona o resultado à lista geral
        resultados.append(r)
        # Incrementa contador caso um protocolo válido tenha sido encontrado
        if r.get("protocolo_17_digitos"):
            encontrados += 1

    # Consolida os resultados no Excel, mantendo histórico acumulativo
    df_total = consolidar_resultados(resultados)
    # Conta o número de protocolos válidos (não nulos) presentes no Excel
    total_ok_excel = int(df_total['protocolo_17_digitos'].replace({"": None}).dropna().shape[0])

    # Cria um resumo com informações gerais do lote processado
    resumo = {
        "total_docs": len(resultados),  # Total de documentos enviados
        "processados": len(resultados),  # Total de documentos processados
        "encontrados": encontrados,  # Quantos documentos continham protocolo válido
        "nao_encontrados": len(resultados) - encontrados,  # Documentos sem protocolo
        "tempo_total_s": round(time.time() - t0, 3)  # Tempo total de processamento em segundos
    }

    # Retorna a resposta em JSON contendo o resumo e os resultados detalhados
    return JSONResponse({
        "resumo": resumo,  # Dados gerais do lote
        "processados": len(resultados),  # Total de documentos processados
        "total_protocolos_no_excel": total_ok_excel,  # Total consolidado no Excel
        "excel_path": "/download/excel",  # Endpoint para download do Excel
        "resultados_lote": resultados  # Resultados individuais de cada documento
    })

# Rota GET para download do arquivo Excel consolidado
@app.get("/download/excel")
def download_excel():
    # Caso o arquivo Excel ainda não exista, retorna erro 404
    if not EXCEL_PATH.exists():
        return JSONResponse({"erro": "Excel ainda não existe"}, status_code=404)
    # Caso exista, retorna o arquivo como resposta para download
    return FileResponse(
        EXCEL_PATH,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # Tipo MIME para Excel
        filename=EXCEL_PATH.name  # Nome do arquivo para download
    )

# Executa a aplicação diretamente via uvicorn quando este arquivo for o ponto de entrada
if __name__ == "__main__":
    # Inicia o servidor na porta 8000, acessível em todas as interfaces (0.0.0.0)
    # reload=True reinicia automaticamente quando há alterações no código
    uvicorn.run("app.main_web:app", host="0.0.0.0", port=8000, reload=True)
