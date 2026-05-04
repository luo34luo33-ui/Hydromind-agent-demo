import re
from pathlib import Path
from typing import Optional


def get_base_dir():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


class CodeTemplateRAG:
    """代码模板检索器 - 根据 Planner 方案匹配最合适的代码模板"""

    KEYWORD_TO_TEMPLATE = {
        "线性水库": ["线性水库", "线性水库模型", "k * S", "慢速响应", "单水箱", "kS", "线性"],
        "Tank Model": ["水箱", "Tank", "三水箱", "多水箱", "串联", "k1", "k2", "k3"],
        "SCS-CN": ["SCS", "CN", "曲线数", "超额蓄水", "SCS"],
        "新安江": ["新安江", "蓄满产流", "流域蒸发", "张力水", "湿润区", "南方", "蓄满"],
        "HBV": ["HBV", "土壤含水量", "上层下层", "rate routine", "温带", "寒带"],
    }

    TEMPLATE_ALIAS = {
        "线性水库": "线性水库模型 (Linear Reservoir)",
        "Tank Model": "Tank Model - 三水箱 (Three-Tank Model)",
        "SCS-CN": "SCS-CN 模型",
        "新安江": "新安江模型 (Xinanjiang Model)",
        "HBV": "HBV 模型 (Hydrological By Växjö)",
    }

    DEFAULT_TEMPLATE = "线性水库模型 (Linear Reservoir)"

    def __init__(self):
        self.templates: dict[str, dict] = {}
        self._load_templates()

    def _load_templates(self):
        """从 hydro_code_templates.md 解析所有模板"""
        doc_path = get_base_dir() / "docs" / "hydro_code_templates.md"
        if not doc_path.exists():
            return

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r'\n##\s+', content)
        for section in sections[1:]:
            model_name = section.split('\n')[0].strip()
            if not model_name:
                continue

            template = self._parse_template_section(section)
            if template:
                self.templates[model_name] = template

    def _parse_template_section(self, section: str) -> Optional[dict]:
        """解析单个模板章节"""
        result = {"description": "", "code": "", "params": {}}

        desc_match = re.search(r'### 适用场景\n(.*?)(?=###|```)', section, re.DOTALL)
        if desc_match:
            result["description"] = desc_match.group(1).strip()

        code_match = re.search(r'```python\n(.*?)```', section, re.DOTALL)
        if code_match:
            result["code"] = code_match.group(1).strip()

        param_pattern = r'\|\s*(\w+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|.*?\|\s*(.*?)\s*\|'
        for match in re.finditer(param_pattern, section):
            param_name = match.group(1)
            desc = match.group(2).strip()
            result["params"][param_name] = desc

        if not result["code"]:
            return None

        return result

    def retrieve_by_plan(self, plan_text: str) -> str:
        """根据 Planner 方案文本自动匹配代码模板"""
        if not plan_text:
            return self._get_template_code(self.DEFAULT_TEMPLATE)

        matched_type = None
        max_score = 0

        plan_lower = plan_text.lower()
        for model_type, keywords in self.KEYWORD_TO_TEMPLATE.items():
            score = sum(1 for kw in keywords if kw.lower() in plan_lower)
            if score > max_score:
                max_score = score
                matched_type = model_type
            if score > max_score:
                max_score = score
                matched_type = model_type

        if matched_type:
            return self._get_template_code(matched_type)

        return self._get_template_code(self.DEFAULT_TEMPLATE)

    def retrieve_by_model_name(self, model_name: str) -> str:
        """直接按模型名称检索"""
        return self._get_template_code(model_name)

    def _get_template_code(self, model_name: str) -> str:
        """获取模板代码，未找到则返回默认"""
        template = self.templates.get(model_name)
        if template:
            return template["code"]
        return self.templates.get(self.DEFAULT_TEMPLATE, {}).get("code", "")

    def get_params_info(self, model_name: str) -> dict:
        """获取模型的参数信息"""
        template = self.templates.get(model_name)
        if template:
            return template.get("params", {})
        return {}

    def get_all_model_names(self) -> list[str]:
        """获取所有可用模型名称"""
        return list(self.templates.keys())


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
