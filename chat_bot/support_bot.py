import os
import random
import logging
from preprocessing import PDFTextExtractor
from indexing import PDFVectorStore
from model import LLMQueryHandler
from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(
    filename="logs.txt",  # Logs will be saved to this file
    level=logging.INFO,   # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

class SupportBotAgent:
    def __init__(self, pdf_path):
        """
        Initializes the AIQuerySystem class by loading API keys, processing the PDF, 
        and setting up the vector store and language model.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logging.error("API key missing! Check your environment variables.")
            raise ValueError("Missing API Key")

        logging.info("API Key Loaded Successfully.")
        self.pdf_path = pdf_path
        self.extractor = PDFTextExtractor(self.pdf_path)
        self.chunks = self.extractor.process_pdf()
        logging.info(f"Loaded document: {self.pdf_path}")

        self.pdf_store = PDFVectorStore()
        self.pdf_store.create_vector_store(self.chunks)
        self.pdf_store.save_vector_store()
        self.vector_store = self.pdf_store.load_vector_store()
        logging.info("Document successfully stored in vector database.")

        self.inference_model = LLMQueryHandler(self.api_key)
        self.llm = LLM(
            model="gpt-4-turbo",
            temperature=0,
            timeout=120,
            max_tokens=4000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            seed=42,
            api_key=self.api_key,
        )

        self.output_validator = Agent(
            role="Response evaluator ensuring high-quality and relevant outputs.",
            goal="Analyze the generated response and provide a rating.",
            backstory="Expert in response validation and relevance assessment.",
            allow_delegation=False,
            llm=self.llm,
            verbose=False
        )

    def train_on_document(self):
        """Trains the AI system by processing and storing the document into a vector store."""
        logging.info("Training complete. Document stored in vector database.")

    def answer_query(self, query):
        """Answers user queries by retrieving relevant documents and generating a response."""
        logging.info(f"Received Query: {query}")

        retrieved_documents = self.vector_store.similarity_search(query, k=5)
        page_contents = [doc.page_content for doc in retrieved_documents]

        response = self.inference_model.generate_response(query, page_contents)
        logging.info(f"Generated Response: {response.content}")

        return response.content, page_contents

    def evaluate_response(self, query, response, context):
        """Evaluates the AI-generated response based on context and query."""
        logging.info(f"Evaluating response for query: {query}")

        output_correction = Task(
            description="Evaluate the response based on context, query, and feedback.",
            expected_output="Evaluation_Result: RATING",
            agent=self.output_validator
        )

        crew = Crew(
            agents=[self.output_validator],
            tasks=[output_correction],
            verbose=False
        )
        inputs = {
            "input_context": context,
            "user_prompt": query,
            "output": response,
        }
        result = crew.kickoff(inputs=inputs)
        evaluation_result = result.raw

        logging.info(f"Evaluation Feedback: {evaluation_result}")
        return evaluation_result

    def adjust_response(self, query, initial_response, context):
        """
        Adjusts the response based on simulated feedback in a loop for up to 2 iterations.
        """
        max_iterations = 2

        for _ in range(max_iterations):
            feedback = random.choice(["not helpful", "too vague", "good"])
            logging.info(f"Feedback: {feedback}")

            if feedback == "good":
                logging.info("Response is acceptable. Returning final response.")
                return initial_response

            elif feedback == "too vague":
                logging.info("Adjusting response by adding more context.")
                context.append("Additional details for clarity.")

            elif feedback == "not helpful":
                logging.info("Adjusting response by rephrasing for better accuracy.")

            initial_response = self.inference_model.generate_response(query, context).content
            logging.info(f"New Adjusted Response: {initial_response}")

        return initial_response


# Example usage
if __name__ == "__main__":
    pdf_path = "dataset/Serri_doc.pdf"
    ai_system = SupportBotAgent(pdf_path)
    ai_system.train_on_document()
    while True:
        user_query = input("Enter your query: ")
        response, context = ai_system.answer_query(user_query)
        evaluation = ai_system.evaluate_response(user_query, response, context)
        adjusted_response = ai_system.adjust_response(user_query, response, context)

        logging.info(f"Final Response: {adjusted_response}")
        print(f"Final Response: {adjusted_response}")















































