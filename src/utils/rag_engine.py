from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from src.templates.registry import TEMPLATE_REGISTRY, ModelTemplate
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    ModelTemplate = None
    TEMPLATE_REGISTRY = {}


def get_base_dir():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


class CodeTemplateRAG:
    """代码模板检索器 - 基于 Python Registry"""

    DEFAULT_TEMPLATE_ID = "linear_reservoir"

    def __init__(self):
        self.templates: Dict[str, Any] = {}
        self._load_templates()

    def _load_templates(self):
        """从 Python Registry 加载模板"""
        if REGISTRY_AVAILABLE:
            self.templates = TEMPLATE_REGISTRY

    def get_templates_by_ids(self, ids: List[str]) -> str:
        """根据 ID 列表获取模板代码 - O(1) 精确寻址"""
        codes = []
        for template_id in ids:
            template = self.templates.get(template_id)
            if template and hasattr(template, 'code'):
                codes.append(template.code)
        
        if not codes:
            print(f"警告: 未匹配到有效模板 ID，使用默认模板")
            default_template = self.templates.get(self.DEFAULT_TEMPLATE_ID)
            if default_template and hasattr(default_template, 'code'):
                return default_template.code
            return ""
        
        return "\n\n".join(codes)

    def retrieve_by_plan(self, plan_text: str) -> str:
        """根据 Planner 方案文本自动匹配代码模板（兼容旧模式）"""
        if not plan_text:
            return self._get_default_code()

        keywords_lower = plan_text.lower()
        best_match = None
        best_score = 0
        
        for template in self.templates.values():
            if not hasattr(template, 'keywords'):
                continue
            score = sum(1 for kw in template.keywords if kw.lower() in keywords_lower)
            if score > best_score:
                best_score = score
                best_match = template
        
        if best_match and hasattr(best_match, 'code'):
            return best_match.code
        
        return self._get_default_code()

    def retrieve_by_model_name(self, model_name: str) -> str:
        """直接按模型名称检索"""
        template = self.templates.get(model_name)
        if template and hasattr(template, 'code'):
            return template.code
        return self._get_default_code()

    def _get_default_code(self) -> str:
        """获取默认模板代码"""
        default_template = self.templates.get(self.DEFAULT_TEMPLATE_ID)
        if default_template and hasattr(default_template, 'code'):
            return default_template.code
        return ""

    def get_params_info(self, template_id: str) -> dict:
        """获取模型的参数信息"""
        template = self.templates.get(template_id)
        if template and hasattr(template, 'params'):
            return template.params
        return {}

    def get_all_model_names(self) -> list[str]:
        """获取所有可用模型名称"""
        return list(self.templates.keys())

    def retrieve_by_keywords(self, keywords: list[str]) -> str:
        """根据关键词列表检索代码模板"""
        if not keywords:
            return self._get_default_code()
        
        keywords_lower = [kw.lower() for kw in keywords]
        
        best_match = None
        best_score = 0
        
        for template in self.templates.values():
            if not hasattr(template, 'keywords'):
                continue
            score = sum(1 for kw in keywords_lower if any(kw in mk.lower() for mk in template.keywords))
            if score > best_score:
                best_score = score
                best_match = template
        
        if best_match and hasattr(best_match, 'code'):
            return best_match.code
        
        return self._get_default_code()


class HydroKnowledgeBase:
    """水文领域 RAG 知识库"""

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
        """检索与 query 最相关的 k 段知识"""
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