import os
import glob
import PyPDF2
import warnings
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import traceback
import requests
from flask import Flask, render_template, request, jsonify
import time
import torch
import psutil

# 設定繁體中文環境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 系統路徑設定
KNOWLEDGE_BASE_PATH = r"C:\rag_system\knowledge_base"
VECTOR_DB_PATH = r"C:\rag_system\vector_db"
os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# 資源監控函數
def check_system_resources():
    """檢查系統資源使用情況"""
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    logger.info(f"系統資源 - 記憶體: {mem.available/1024**3:.1f}GB 可用 / CPU 使用率: {cpu}%")
    return {
        'total_ram': mem.total/1024**3,
        'available_ram': mem.available/1024**3,
        'cpu_cores': psutil.cpu_count(logical=False)
    }

# 初始化設定
system_resources = check_system_resources()

# 模型選擇邏輯 (針對20GB RAM優化)
def select_model_for_20gb():
    """為20GB RAM系統選擇合適模型"""
    models_to_try = [
        ("qwen:4b", 7),    # 通義千問7B繁體中文模型
        ("llama3:8b", 8),  # Meta Llama3 8B
        ("openchat:7b", 7) # OpenChat 7B
    ]
    
    for model, size in models_to_try:
        if check_ollama_model(model):
            if system_resources['available_ram'] > size * 1.5:  # 保留1.5倍緩衝
                logger.info(f"✅ 選擇模型: {model} (需約{size}GB RAM)")
                return model
    logger.error("❌ 沒有找到合適的模型，請安裝至少一個7B以下模型")
    exit(1)

# Ollama模型檢查
def check_ollama_model(model_name):
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"model": model_name},
            timeout=15
        )
        if response.status_code == 200:
            logger.info(f"模型可用: {model_name}")
            return True
        return False
    except Exception as e:
        logger.error(f"檢查模型時出錯: {str(e)}")
        return False

# 初始化ChromaDB
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# 選擇模型
MODEL_NAME = select_model_for_20gb()

# 設定Ollama參數 (記憶體優化版)
ollama_params = {
    "model": MODEL_NAME,
    "request_timeout": 300,
    "additional_kwargs": {
        "options": {
            "num_ctx": 2048,      # 減少上下文長度
            "num_thread": 6,       # 限制執行緒數
            "num_gpu": -1,         # 自動GPU加速
            "low_vram": True       # 低顯存模式
        }
    }
}
Settings.llm = Ollama(**ollama_params)

# 繁體中文嵌入模型 (記憶體優化)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"  # 基礎版模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
    embed_batch_size=24  # 較小的批次大小
)

# 向量資料庫設定
try:
    chroma_client.delete_collection("knowledge_base")
    logger.info("清理現有知識庫集合")
except:
    logger.info("無需清理，開始新建集合")

chroma_collection = chroma_client.create_collection(
    name="knowledge_base",
    metadata={
        "hnsw:space": "cosine",  # 中文適合餘弦相似度
        "dimension": 768          # 基礎模型維度
    }
)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 記憶體優化的PDF處理
def load_pdfs_with_memory_control(folder_path, max_files=15, max_pages=100):
    """帶記憶體控制的PDF讀取函數"""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))[:max_files]
    documents = []
    metadatas = []
    
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                
                for i, page in enumerate(reader.pages[:max_pages]):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
                    
                    # 每處理10頁檢查一次記憶體
                    if i % 10 == 0:
                        mem = psutil.virtual_memory()
                        if mem.available < 2 * 1024**3:  # 低於2GB時警告
                            logger.warning(f"記憶體不足，停止處理 {pdf_file}")
                            break
                
                if text.strip():
                    documents.append(text)
                    metadatas.append({
                        "source": os.path.basename(pdf_file),
                        "pages_processed": min(len(reader.pages), max_pages)
                    })
                    logger.info(f"已載入: {os.path.basename(pdf_file)} ({len(text)}字)")
                    
        except Exception as e:
            logger.error(f"處理 {pdf_file} 時出錯: {str(e)}")
    
    return documents, metadatas

# 繁體中文文本分割
def split_chinese_text(documents, metadatas):
    """專門處理繁體中文的文本分割"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,           # 較小的塊大小
        chunk_overlap=70,         # 適當重疊
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "…", "，", "、", " "]
    )
    
    all_chunks = []
    all_metadatas = []
    
    for doc, meta in zip(documents, metadatas):
        chunks = text_splitter.split_text(doc)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            new_meta = meta.copy()
            new_meta.update({
                "chunk_id": i,
                "char_count": len(chunk)
            })
            all_metadatas.append(new_meta)
    
    logger.info(f"文本分割完成: 共 {len(all_chunks)} 個文本塊")
    return all_chunks, all_metadatas

# 記憶體友好的索引構建
def build_index_with_memory_control(chunks, metadatas):
    """帶記憶體控制的索引構建"""
    from llama_index.core.schema import TextNode
    
    nodes = []
    for idx, (text, meta) in enumerate(zip(chunks, metadatas)):
        nodes.append(TextNode(text=text, metadata=meta))
        
        # 每處理50個節點檢查記憶體
        if idx % 50 == 0:
            mem = psutil.virtual_memory()
            if mem.available < 3 * 1024**3:  # 低於3GB時暫停
                logger.warning("記憶體不足，暫停處理進行回收...")
                import gc
                gc.collect()
                time.sleep(2)
    
    try:
        # 分批建立索引
        batch_size = 100
        index = None
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            if index is None:
                index = VectorStoreIndex(
                    batch,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model,
                    show_progress=True
                )
            else:
                index.insert_nodes(batch)
            
            logger.info(f"已處理 {min(i+batch_size, len(nodes))}/{len(nodes)} 個節點")
            
        return index
    except Exception as e:
        logger.error(f"建立索引失敗: {str(e)}")
        return None

# 繁體中文查詢模板
TRADITIONAL_CHINESE_PROMPT = PromptTemplate(
    "你是一個專門回答繁體中文問題的AI助手，請根據以下上下文用臺灣常用的正體中文回答問題。\n"
    "上下文:\n{context_str}\n"
    "問題: {query_str}\n"
    "請注意:\n"
    "1. 使用正體中文回答\n"
    "2. 使用臺灣常用術語\n"
    "3. 回答要完整準確\n"
    "4. 如果不知道答案請誠實說明\n"
    "回答:"
)

# 查詢引擎設定
def create_query_engine(index):
    """建立記憶體優化的查詢引擎"""
    return index.as_query_engine(
        similarity_top_k=2,  # 減少檢索結果數量
        text_qa_template=TRADITIONAL_CHINESE_PROMPT,
        streaming=False,
        node_postprocessors=[]
    )

# 初始化Flask應用
app = Flask(__name__)
index = None

# 系統初始化
def initialize_system():
    global index
    logger.info("====== 繁體中文RAG系統初始化 ======")
    
    # 1. 載入PDF
    logger.info("階段1: 載入PDF文件...")
    docs, metas = load_pdfs_with_memory_control(KNOWLEDGE_BASE_PATH)
    if not docs:
        logger.error("沒有載入任何文件，請檢查knowledge_base資料夾")
        return False
    
    # 2. 處理文本
    logger.info("階段2: 處理繁體中文文本...")
    chunks, chunk_metas = split_chinese_text(docs, metas)
    
    # 3. 建立索引
    logger.info("階段3: 建立向量索引...")
    index = build_index_with_memory_control(chunks, chunk_metas)
    
    if not index:
        logger.error("索引建立失敗")
        return False
    
    logger.info("✅ 系統初始化完成")
    return True

# Flask路由
@app.route('/')
def home():
    return render_template('index_zh_tw.html',
                         model_name=MODEL_NAME,
                         ram_available=f"{system_resources['available_ram']:.1f}GB")

@app.route('/query', methods=['POST'])
def handle_query():
    if not index:
        return jsonify({"error": "系統未初始化"}), 503
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "無效請求"}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({"error": "查詢內容為空"}), 400
    
    try:
        start_time = time.time()
        query_engine = create_query_engine(index)
        response = query_engine.query(query)
        elapsed = time.time() - start_time
        
        logger.info(f"查詢完成 - 耗時: {elapsed:.2f}s - 查詢: '{query}'")
        return jsonify({
            "response": str(response),
            "time": f"{elapsed:.2f}s"
        })
    except Exception as e:
        logger.error(f"查詢處理錯誤: {str(e)}")
        return jsonify({"error": "處理查詢時出錯"}), 500

# 繁體中文HTML模板
def create_chinese_template():
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    with open(os.path.join(template_dir, 'index_zh_tw.html'), 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>繁體中文知識庫系統</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Microsoft JhengHei', Arial, sans-serif;
        }
        body {
            background-color: #f0f4f8;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 25px;
        }
        header {
            text-align: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        h1 {
            color: #2b6cb0;
            margin-bottom: 10px;
        }
        .sys-info {
            background: #ebf8ff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .info-row {
            display: flex;
            margin-bottom: 8px;
        }
        .info-label {
            font-weight: bold;
            min-width: 120px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 2px solid #cbd5e0;
            border-radius: 6px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background: #2b6cb0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
            transition: background 0.3s;
        }
        button:hover {
            background: #2c5282;
        }
        #response {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
            min-height: 150px;
            white-space: pre-wrap;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid rgba(0,0,0,0.1);
            border-radius: 50%;
            border-top: 4px solid #2b6cb0;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>繁體中文知識庫系統</h1>
            <p>基於大型語言模型的智能問答系統</p>
        </header>
        
        <div class="sys-info">
            <div class="info-row">
                <span class="info-label">當前模型:</span>
                <span id="model">{{ model_name }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">可用記憶體:</span>
                <span id="memory">{{ ram_available }}</span>
            </div>
        </div>
        
        <textarea id="query" placeholder="請輸入您的繁體中文問題..."></textarea>
        <button onclick="submitQuery()">提交查詢</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>處理中，請稍候...</p>
        </div>
        
        <div id="response">請輸入問題後點擊「提交查詢」按鈕</div>
    </div>

    <script>
        function submitQuery() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('請輸入查詢內容！');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('response').textContent = '';
            
            fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                if (data.error) {
                    document.getElementById('response').textContent = '錯誤: ' + data.error;
                } else {
                    document.getElementById('response').textContent = data.response;
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('response').textContent = '查詢失敗: ' + error;
            });
        }
    </script>
</body>
</html>''')

# 主程序
if __name__ == "__main__":
    create_chinese_template()
    
    if initialize_system():
        logger.info("啟動伺服器... 訪問 http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("系統初始化失敗，請檢查日誌")
