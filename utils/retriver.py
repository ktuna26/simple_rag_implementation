from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import


class Retriever:
    def __init__(self, index_path: str, embedding_model_name: str, llm_model_name: str):
        """
        Initializes the Retriever class with the necessary parameters.
        
        Args:
            index_path (str): Path to the saved FAISS vector store index.
            embedding_model_name (str): Name of the embedding model used to encode documents and queries.
            llm_model_name (str): Name of the Hugging Face model used for language generation.
        """
        self.index_path = index_path  # Path to the FAISS vector store
        self.embedding_model_name = embedding_model_name  # Embedding model for creating document vectors
        self.llm_model_name = llm_model_name  # Language model for generating answers

    def load_vectorstore(self):
        """
        Load the FAISS vector store from the specified path.
        
        This method loads the pre-built FAISS index and initializes the embeddings for queries.
        
        Returns:
            FAISS: The loaded FAISS vector store.
        """
        # Load the FAISS index using the provided embedding model
        return FAISS.load_local(
            self.index_path, 
            HuggingFaceEmbeddings(model_name=self.embedding_model_name),
            allow_dangerous_deserialization=True  # Allow deserialization of potentially unsafe pickle files
        )

    def initialize_llm(self, max_length=100):  # Default max_length of 100 tokens
        """
        Initialize and load the Hugging Face language model for text generation.
        
        This method creates a pipeline for text generation using the specified model.
        
        Args:
            max_length (int): The maximum length of the generated text (in tokens).
            
        Returns:
            HuggingFacePipeline: The initialized language model wrapped in a HuggingFacePipeline object.
        """
        # Initialize the text generation pipeline from Hugging Face with the max_length parameter
        llm_pipeline = pipeline(
            "text-generation", 
            model=self.llm_model_name, 
            device=0,  # device=0 for MPS
            max_length=max_length,  # Control the max length of the generated text
            truncation=True,  # Explicitly enable truncation
            pad_token_id=50256  # Set pad_token_id (50256 is often used for GPT-based models like GPT-2 or GPT-3)
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)

    def build_retriever(self, vectorstore):
        """
        Build the retrieval-based question-answering (QA) chain using the FAISS vector store.
        
        Args:
            vectorstore (FAISS): The FAISS vector store containing the document embeddings.
        
        Returns:
            RetrievalQA: A retrieval-based QA chain that uses the loaded vector store and language model.
        """
        # Convert the vectorstore into a retriever for document search
        retriever = vectorstore.as_retriever()
        
        # Initialize the language model
        llm = self.initialize_llm(max_length=200)
        
        # Build and return the QA chain using the retriever and language model
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    def run(self):
        """
        Run the retriever pipeline in a loop, allowing the user to ask questions.
        
        This method continuously prompts the user for queries, retrieves relevant documents from the FAISS index, 
        and generates answers using the language model. The source documents are also returned for reference.
        """
        # Load the FAISS vector store
        vectorstore = self.load_vectorstore()
        
        # Build the QA chain with the vector store and language model
        qa_chain = self.build_retriever(vectorstore)

        # Start the loop where the user can input questions
        while True:
            query = input("Enter your question (type 'exit' to quit): ")
            if query.lower() == "exit":  # Exit condition
                break

            # Get the answer from the QA chain based on the query
            result = qa_chain.invoke({"query": query})
            
            # Output the generated answer and the source documents
            print("\nAnswer:", result["result"])
            print("Source Documents:", result["source_documents"])