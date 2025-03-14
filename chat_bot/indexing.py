from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PDFVectorStore:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
        """Initialize the vector store with an embedding model."""
        self.model_name = model_name
        self.hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.vectorstore = None

    def format_chunks_for_output(self, chunks, source_file="Serri_doc.pdf"):
        """Convert chunks into LangChain Document format."""
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append(Document(
                page_content=f"Title: {chunk['title']}\n\n{chunk['content']}",
                metadata={"source": source_file, "chunk": i + 1}
            ))
        return formatted_chunks

    def create_vector_store(self, chunks, source_file="Serri_doc.pdf"):
        """Create a FAISS vector store from document chunks."""
        formatted_chunks = self.format_chunks_for_output(chunks, source_file)
        self.vectorstore = FAISS.from_documents(formatted_chunks, self.hf)
        return self.vectorstore

    def save_vector_store(self, path="serri_doc"):
        """Save the vector store to a local file."""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vector_store(self, path="serri_doc"):
        """Load the vector store from a local file."""
        self.vectorstore = FAISS.load_local(path, self.hf, allow_dangerous_deserialization=True)
        return self.vectorstore


# # Example Usage
# if __name__ == "__main__":
#     chunks = [
#         {"title": "Overview of Serri AI", "content": "Serri AI is a growth engine..."},
#         {"title": "Mission", "content": "Serri AI's core mission is to empower SMEs..."}
#     ]
    
#     pdf_store = PDFVectorStore()
#     pdf_store.create_vector_store(chunks)
#     pdf_store.save_vector_store()
#     new_vector_store = pdf_store.load_vector_store()
