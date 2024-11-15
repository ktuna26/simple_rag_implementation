from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class Data:
    def __init__(self, embedding_model_name: str, knowledge_base_path: str, index_save_path: str):
        """
        Initializes the Data class with the necessary parameters.
        
        Args:
            embedding_model_name (str): The name of the model to use for generating embeddings.
            knowledge_base_path (str): Path to the knowledge base (text file) to be processed.
            index_save_path (str): Directory where the FAISS vector store will be saved.
        """
        self.embedding_model_name = embedding_model_name  # The embedding model to be used (HuggingFace)
        self.knowledge_base_path = knowledge_base_path  # Path to the knowledge base file
        self.index_save_path = index_save_path  # Directory to save the FAISS index
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)  # Instantiate embeddings

    def process_knowledge_base(self):
        """
        Load the knowledge base document, split it into chunks, and return the chunks.
        
        This method loads the document using TextLoader, and then splits the document into 
        smaller chunks to optimize processing and indexing.
        
        Returns:
            List[Document]: List of split documents.
        """
        loader = TextLoader(self.knowledge_base_path)  # Load the knowledge base text file
        documents = loader.load()  # Read the contents of the knowledge base file

        # Split the loaded documents into chunks of 1000 characters with an overlap of 200 characters
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        return text_splitter.split_documents(documents)  # Return the split text chunks

    def create_vectorstore(self, texts):
        """
        Create a FAISS vector store from the provided texts and save it locally.
        
        Args:
            texts (List[Document]): A list of processed document chunks to be indexed.
        """
        # Create the FAISS vector store from the provided texts using the embedding model
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        # Save the vector store to the specified local path
        vectorstore.save_local(self.index_save_path)
        
        # Print a message confirming the save location
        print(f"Vector store saved to {self.index_save_path}")

    def run(self):
        """
        Execute the entire data pipeline:
        1. Process th knowledge base into document chunks.
        2. Create and save the FAISS vector store with the chunks.
        """
        # Process the knowledge base into smaller chunks
        texts = self.process_knowledge_base()
        
        # Create and save the vector store using the processed chunks
        self.create_vectorstore(texts)