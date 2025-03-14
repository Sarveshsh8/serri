from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage



class LLMQueryHandler:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4-turbo"
        )
        self.template = """Question: {question}\n\nAnswer: Let's think step by step."""
        self.prompt = PromptTemplate.from_template(self.template)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def query(self, question: str) -> str:
        return self.llm_chain.run(question)

    def generate_response(self, text: str, context: str) -> str:
        """Generate a response from the model based on query and context."""
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=f"Query: {text}\nContext: {context}\nProvide only the answer.")
        ]
        return self.llm.invoke(messages)  # ✅ Corrected input format



