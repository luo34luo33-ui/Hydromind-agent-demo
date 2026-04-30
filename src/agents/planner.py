from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


PLANNER_SYSTEM_PROMPT = """你是一位资深水文学家，擅长根据流域特征设计概念性水文模型模拟方案。

你的任务：
1. 分析给定的流域属性（面积、坡度、渗透性、气候类型等）
2. 判定该流域的水文响应特征（快速/中等/慢速响应）
3. 提出一个产流+汇流模拟方案，推荐合适的模型参数

原则：
- 大面积 + 低坡度 + 干旱 → 慢速响应，适合低出流系数 + 较大储水容量
- 小面积 + 高坡度 + 湿润 → 快速响应，适合高出流系数 + 较小储水容量
- 中等面积 + 中等坡度 → 中等响应
- 气候湿润 → 蒸散发量较大
- 气候干旱 → 蒸散发量较小但降雨稀少

输出格式：
先给出 1-2 句总体判断，然后列出具体的产流方案和汇流方案，最后给出建议参数值。"""


class Planner:

    def __init__(self, openai_api_key, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=openai_api_key,
        )

    def plan(self, attributes, context, user_request=""):
        extra = ""
        if user_request:
            extra = (
                f"\n用户额外需求（务必遵守）:\n{user_request}\n"
            )
        user_message = (
            f"流域特征: {attributes}\n"
            f"参考知识:\n{context}\n"
            f"{extra}"
            f"请根据以上信息，提出产流+汇流模拟方案，并推荐模型参数。"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("user", "{user_message}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_message": user_message})
        return response.content
