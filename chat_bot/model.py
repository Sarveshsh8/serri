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



# model list - HuggingFaceTB/SmolLM-1.7B-Instruct, M4-ai/TinyMistral-6x248M-Instruct, HuggingFaceTB/SmolLM2-135M, Qwen/Qwen2-0.5B
# class SmolLM2Inference:
#     def __init__(self, checkpoint="Qwen/Qwen2.5-0.5B", device="cpu"):
#         """Initialize the model and tokenizer."""
#         self.checkpoint = checkpoint
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

#         # Ensure PAD token is set correctly
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

#     def generate_response(self, text, context):
#         """Generate a response from the model based on query and context."""
#         prompt = """
#         You are a helpful AI assistant who is good at answering queries!
#         I will be providing you the context data also to help you generate a clear and concise answer.

#         Query: {text}
#         Context: {context}
#         You are supposed to provide only the answer
#         """
        
#         formatted_prompt = prompt.format(text=text, context=context[0])
#         messages = [{"role": "user", "content": formatted_prompt}]

#         start_time = time.time()
#         input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

#         # Encode input with attention_mask
#         inputs = self.tokenizer.encode(
#             input_text, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True,
#             max_length=4096, 
#             # attention_mask=True
#         ).to(self.device)

#         output_embeds = self.model.generate(
#             inputs, 
#             max_new_tokens=100, 
#             temperature=0.1, 
#             top_p=0.9, 
#             do_sample=True
#         )[0][len(inputs[0]):]
#         output = self.tokenizer.decode(output_embeds, skip_special_tokens=True)
#         end_time = time.time()

#         print("Time taken:", end_time - start_time)
#         return output

# # Example usage
# if __name__ == "__main__":
#     inference_model = SmolLM2Inference()
#     query = "what is the weather in Bangalore today?"
#     response = inference_model.generate_response(query)
#     print(response)
