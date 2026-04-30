from pathlib import Path


def get_base_dir():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


class HydroKnowledgeBase:
    """
    水文领域 RAG 知识库。

    优先使用本地 Chroma 向量数据库检索；
    若加载失败，自动降级为关键词匹配检索。
    """

    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.vectorstore = None
        self.chunks = []
        self.index_built = False
        self.index_attempted = False
        self._load_chunks()

    def _load_chunks(self):
        """预先加载文档分块（供降级检索使用）"""
        docs_dir = get_base_dir() / "docs"
        doc_path = docs_dir / "hydro_knowledge.md"

        if not doc_path.exists():
            return

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n", " "],
        )
        self.chunks = splitter.split_text(text)

    def _build_index(self):
        """加载本地 Chroma 向量库"""
        if self.index_attempted:
            return
        self.index_attempted = True

        try:
            from langchain_community.vectorstores import Chroma
            
            vector_db_dir = get_base_dir() / "vector_db"
            if not vector_db_dir.exists():
                self.vectorstore = None
                return

            self.vectorstore = Chroma(
                persist_directory=str(vector_db_dir),
                embedding_function=None
            )
            self.index_built = True
        except Exception as e:
            self.vectorstore = None

    def retrieve(self, query, k=3):
        """
        检索与 query 最相关的 k 段知识。
        向量检索失败时自动降级为关键词匹配。
        """
        if not self.chunks:
            return "(知识库文档不存在)"

        self._build_index()

        if self.vectorstore is not None:
            try:
                docs = self.vectorstore.similarity_search(query, k=k)
                return "\n\n".join(doc.page_content for doc in docs)
            except Exception:
                pass

        return self._keyword_retrieve(query, k)

    def _keyword_retrieve(self, query, k=3):
        """关键词匹配降级检索"""
        keywords = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            lower_chunk = chunk.lower()
            score = sum(1 for kw in keywords if kw in lower_chunk)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for s, c in scored[:k] if s > 0]
        if not top:
            top = self.chunks[:k]
        return "\n\n".join(top)
