import warnings
from utils import Data, Retriever

warnings.filterwarnings("ignore", message="To copy construct from a tensor*")


def main():
    # Step 1: Initialize and run the data pipeline
    # This pipeline will process the knowledge base and create a vector store for efficient retrieval.
    
    data_pipeline = Data(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # The embedding model used for vectorizing the text
        knowledge_base_path="data/texts/knowledge_base.txt",  # Path to the knowledge base file
        index_save_path="faiss_index"  # Directory to save the FAISS index (used for similarity search)
    )
    
    # Inform the user that the knowledge base is being processed and the vector store is being created
    print("Processing knowledge base and creating vector store...")
    
    # Running the data pipeline to process the knowledge base and store embeddings
    data_pipeline.run()

    # Step 2: Initialize and run the retriever pipeline
    # This pipeline will load the pre-processed index and use a language model for question-answering.
    
    retriever_pipeline = Retriever(
        index_path="faiss_index",  # Path to the pre-built FAISS index
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # The embedding model to use for encoding the queries
        llm_model_name="google/flan-t5-base"  # Language model used for generating answers from the retrieved documents
    )
    
    # Inform the user that the question-answering loop is starting
    print("Starting question-answering loop...")
    
    # Running the retriever pipeline to process user queries and provide answers
    retriever_pipeline.run()

# Ensure the main function runs when this script is executed directly
if __name__ == "__main__":
    main()