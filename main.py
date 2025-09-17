"""
    This is the entry point of the RAG enhancement project. It integrates the LLM backend
    (from llm.py) and the GUI (from llm_gui.py) to provide a complete application workflow.
    The main script coordinates user interaction, query processing, and FAQ updates.

"""

import tkinter as tk
from tkinter import ttk
from llm import LangModel
from llm_gui import ChatbotGUI
   
# ---------------- Run the App ----------------
if __name__ == "__main__":

    root = tk.Tk()
    chatbot = LangModel()
    app = ChatbotGUI(root, chatbot)

    exit_button = ttk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=5)

    root.mainloop()

