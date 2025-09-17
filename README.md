LLM-RAG Enhancement Project

This project is an AI-powered chatbot built using Large Language Models (LLMs) and an enhanced Retrieval-Augmented Generation (RAG) approach.
The system improves chatbot accuracy by maintaining a knowledge base, vector database, and FAQ database while continuously learning from user interactions.


⚙️ Features

Chatbot with RAG approach – retrieves context from knowledge base and FAQ database.

FAQ auto-update mechanism – chatbot responses are stored in the FAQ DB for future reuse.

Vector database integration – supports semantic similarity search using embeddings.

Clean code structure – object-oriented programming with documentation.

GUI support – user-friendly interface via llm_gui.py.


🚀 Installation

Clone the repository:

git clone https://github.com/anakagzo/LLM-RAG-enhancement-project.git
cd LLM-RAG-enhancement-project


Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Create a .env file in the project root and add your API keys :

OPENAI_API_KEY=your_api_key_here


▶️ Usage

Run the chatbot:

python main.py



📖 Requirements

Python 3.9+

Dependencies listed in requirements.txt

API key (e.g., OpenAI) 

🧑‍💻 Author

Developed by anakagzo (Ugochukwu)
 as part of an academic placement project.
