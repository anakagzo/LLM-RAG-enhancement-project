"""
    This module defines the LangModel class and related functionality for the RAG enhancement project.
    It provides methods for interacting with the vector databases, including updating the FAQ
    database, retrieving context, and managing embeddings for improved chatbot performance.
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough


class LangModel:
    """A class to manage document ingestion, vector storage, retrieval,
    and querying with LLM using LangChain + ChromaDB.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", base_dir: str = ".", db_name: str = "vector_db", faq_db_name: str = "faq_db"):
        """Initialize the LangModel.

        Args:
            model_name (str): OpenAI model name to use.
            db_name (str): Name of the main vector database.
            faq_db_name (str): Name of the FAQ vector database.
        """
        load_dotenv(override=True)
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

        self.model_name = model_name
        self.base_dir = base_dir
        self.db_name = db_name
        self.faq_db_name = faq_db_name
        self.folders = glob.glob("knowledge_base/*")

        self.embeddings = OpenAIEmbeddings()
        self.callback_handler = StdOutCallbackHandler()
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0, callbacks=[self.callback_handler])

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a knowledgeable assistant for YesLove, specializing in providing accurate information.
            Please keep your responses brief and precise. If you dont know the answer say so - don't speculate.
            Provide answers only from the provided context. If there is any contradiction, 
            prioritize information under 'PRIORITY FACTS' and provide links for sources when available.
            

            Context:
            {context}

            Question: {question}
            Answer:
            """
        )
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
        )

        # Initialize vector DBs
        self.vector_stores = {}
        self.retrievers = {}
        self._initialize_vector_dbs()

        # Ensure fresh DB - uncomment to reset DB each time
        #if os.path.exists(self.db_name):
            #Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()

    def _initialize_vector_dbs(self) -> None:
        """Load or create both knowledge base and FAQ vector databases."""
        # Knowledge base DB
        if not os.path.exists(self.db_name):
            self.vector_stores["vector_db"] = self._create_vector_db()
        else:
            self.vector_stores["vector_db"] = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
        self.retrievers["vector_db"] = self.vector_stores["vector_db"].as_retriever(search_kwargs={"k": 5})

        # FAQ DB
        if not os.path.exists(self.faq_db_name):
            self.vector_stores["faq_db"] = Chroma(embedding_function=self.embeddings, persist_directory=self.faq_db_name)
        else:
            self.vector_stores["faq_db"] = Chroma(persist_directory=self.faq_db_name, embedding_function=self.embeddings)      
        self.retrievers["faq_db"] = self.vector_stores["faq_db"].as_retriever(search_kwargs={"k": 3})     
        print("Vectorstores created") 

    def _add_metadata(self, doc: Document, doc_type: str) -> Document:
        """Add metadata to a document.

        Args:
            doc (Document): LangChain document.
            doc_type (str): Type/category of the document.

        Returns:
            Document: Document with added metadata.
        """
        doc.metadata["doc_type"] = doc_type
        return doc

    def _create_chunks(self):
        """Load documents from folders and split them into chunks.

        Returns:
            list[Document]: List of chunked documents.
        """
        text_loader_kwargs = {'encoding': 'utf-8'}
        documents = []

        for folder in self.folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            documents.extend([self._add_metadata(doc, doc_type) for doc in folder_docs])

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        print(f"Total number of chunks: {len(chunks)}")
        return chunks

    def _create_vector_db(self):
        """Create or load vector databases (main + FAQ).

        Returns:
            dict: Dictionary with 'vector_store' and 'faq' vector databases.
        """
        chunks = self._create_chunks()

        vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.db_name)

        return vectorstore

    def reload_faq_db(self):
        """Reload only the FAQ vector database."""
        faq_path = os.path.join(self.base_dir, "faq_db")
        self.vector_stores["faq_db"] = Chroma(persist_directory=faq_path, embedding_function=self.embeddings)
        self.retrievers["faq_db"] = self.vector_stores["faq_db"].as_retriever(search_kwargs={"k": 3})

    def get_combined_context(self, query: str):
        """Retrieve and combine context from both databases.

        Args:
            query (str): User query.

        Returns:
            tuple: (combined_context, combined_docs)
        """
        
        docs1 = self.retrievers['faq_db'].get_relevant_documents(query)
        docs2 = self.retrievers['vector_db'].get_relevant_documents(query)
        combined_docs = docs1 + docs2
        combined_context = "\n".join([doc.page_content for doc in combined_docs])
        return combined_context, combined_docs

    def get_combined_context_with_scores(self, query: str):
        """Retrieve and combine context with similarity scores.

        Args:
            query (str): User query.

        Returns:
            tuple: (combined_context, scored_docs)
        """
        docs_scores1 = self.vector_stores['faq_db'].similarity_search_with_score(query, k=3)
        docs_scores2 = self.vector_stores['vector_db'].similarity_search_with_score(query, k=5)
        combined = docs_scores1 + docs_scores2
        scored_docs = []

        for doc, score in combined:
            doc.metadata["similarity_score"] = score
            scored_docs.append(doc)

        combined_context = "\n".join(
            [f"[Score={doc.metadata['similarity_score']:.4f}] {doc.page_content}" for doc in scored_docs]
        )
        return combined_context

    def query_chatbot(self, user_query: str):
        """Query the LLM with combined context.

        Args:
            user_query (str): Query string.

        Returns:
            dict: LLM response.
        """
        context = self.get_combined_context_with_scores(user_query)
        #print("\n==== QUERY SENT TO LLM ====")
        #print(user_query)
        print("\n==== CONTEXT SENT TO LLM ====")
        print(context)

        response = self.chain.invoke({"context": context, "question": user_query},)
                                     #config={"callbacks": [self.callback_handler]})
        self.add_faq_document(user_query, response.content)
        return response

    def add_faq_document(self, query: str, response: str):
        """Store chatbot response as FAQ document if not duplicate.
        Also allow user to create FAQ document

        Args:
            query (str): User query.
            response (dict): LLM response.
        """
        text = 'PRIORITY FACTS (start):\n' + response + '\t (end facts)'
        metadata = {"source": "FAQ", 'question': query}

        faq = self.vector_stores['faq_db']
        similar_docs_with_score = faq.similarity_search_with_score(response, k=1)

        if not similar_docs_with_score: # is there document in the db
            chunk = Document(page_content=text, metadata=metadata)
            faq.add_documents([chunk])
            self.reload_faq_db()
            return "FAQ added and DB refreshed"
        
        doc, distance = similar_docs_with_score[0]
        similarity = 1 - distance

        if similarity >= 0.8:
            return "Semantically duplicate, skipping: " + query[:50]
        else:
            chunk = Document(page_content=text, metadata=metadata)
            faq.add_documents([chunk])
            self.reload_faq_db()
            return "FAQ added and DB refreshed"

    def _get_faq_document(self, doc_id: str):
        """Retrieve a document from the FAQ database.

        Args:
            doc_id (str): Unique ID of the document.

        Returns:
            tuple: (doc_id, doc_question, faq)
        """
        faq = self.vector_stores['faq_db']
        doc = faq._collection.get(ids=[doc_id])  

        if not doc['ids']:
            raise ValueError(f"No document found with id {doc_id}")

        doc_question = doc['metadatas'][0]['question']
        return doc_id, doc_question, faq


    def update_faq_document(self, updated_text: str, doc_id: str):
        """Update a document in the FAQ database.

        Args:
            updated_text (str): New content to replace old FAQ.
            doc_id (str): Unique ID of document to update.
        """
        doc_id, doc_question, faq = self._get_faq_document(doc_id)
        text = 'PRIORITY FACTS (start):\n' + updated_text + '\t (end facts)'
        metadata = {'source': 'FAQ', 'question': doc_question}
        doc_to_update = Document(page_content=text, metadata=metadata)

        faq.update_document(doc_id, doc_to_update)
        self.reload_faq_db()
        return "FAQ updated and DB refreshed"


    def delete_faq_document(self, doc_id: str):

        """
        Delete a FAQ document from the FAQ vector DB by its ID.
        After deletion, reloads the FAQ DB to keep it in sync.

        Args:
            doc_id (str): The ID of the document to delete.
        """
        if not self.vector_stores['faq_db']:
            raise ValueError("FAQ store not initialized.")

        try:
            self.vector_stores['faq_db']._collection.delete(ids=[doc_id])
            # refresh FAQ DB after deletion
            self.reload_faq_db()
            return f"✅ Deleted FAQ document with ID: {doc_id}"
        except Exception as e:
            return f"⚠️ Failed to delete FAQ document {doc_id}: {e}"

    def list_faq_documents(self) -> list[dict]:
        """
        Retrieve all FAQ documents with their IDs, questions, and page content.

        Returns:
            list[dict]: A list of dictionaries containing id, question, and page_content.
        """
        if "faq_db" not in self.vector_stores:
            self.reload_faq_db()

        # Access underlying collection
        collection = self.vector_stores["faq_db"]._collection
        results = collection.get(include=["documents", "metadatas"])

        faq_list = []
        for idx, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][idx] or {}
            question = metadata.get("question", None)

            faq_list.append({
                "id": results["ids"][idx],
                "question": question,
                "page_content": doc
            })

        return faq_list
    
