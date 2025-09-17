"""
    This module implements the graphical user interface (GUI) for the RAG enhancement project.
    It allows users to interact with the system in a more intuitive way, providing a front-end 
    for querying the chatbot, visualising results, and monitoring updates to the FAQ database.
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from langchain.callbacks.base import BaseCallbackHandler
from llm import LangModel


# --- Custom callback that writes tokens into Tkinter text widget ---
class TkinterCallbackHandler(BaseCallbackHandler):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def on_llm_new_token(self, token: str, **kwargs):
        # Append token to the output text widget
        self.text_widget.insert(tk.END, token)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()


class ChatbotGUI:
    def __init__(self, root, lang_model: LangModel):
        self.root = root
        self.root.title("Chatbot GUI")

        self.lang_model = lang_model

        # Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        # Create frames for chat and admin
        self.chat_frame = ttk.Frame(self.notebook)
        self.admin_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_frame, text="Chat")
        self.notebook.add(self.admin_frame, text="Admin")

        self.build_chat_frame()
        self.build_admin_frame()

    # ---------------- Chat Screen ----------------
    def build_chat_frame(self):
        # --- In build_chat_frame ---
        self.query_entry = ttk.Entry(self.chat_frame)
        self.query_entry.pack(fill="x", padx=10, pady=5, expand=True)

        query_button = ttk.Button(self.chat_frame, text="Ask", command=self.ask_query)
        query_button.pack(pady=5)

        self.response_box = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD)
        self.response_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure tag for bold text
        self.response_box.tag_configure("bot", font=("TkDefaultFont", 10, "bold"))


    def ask_query(self):
        query = self.query_entry.get()
        if not query.strip():
            messagebox.showwarning("Warning", "Please enter a query")
            return

        # Insert user query
        self.response_box.insert(tk.END, f"\nUser: {query}\n")
        # Insert bot label in bold
        self.response_box.insert(tk.END, "Bot: ", "bot")

        # Run the chain
        response = self.lang_model.query_chatbot(query)

        # Insert bot response in bold
        self.response_box.insert(tk.END, f"{response.content}\n\n", "bot")
        self.response_box.see(tk.END)
        self.query_entry.delete(0, tk.END)

        

    # ---------------- Admin Screen ----------------
    def build_admin_frame(self):
        top_frame = ttk.Frame(self.admin_frame)
        top_frame.pack(fill="x", pady=5)

        refresh_button = ttk.Button(top_frame, text="Refresh", command=self.refresh_faq_list)
        refresh_button.pack(side="left", padx=5)

        add_button = ttk.Button(top_frame, text="Add FAQ", command=self.add_faq)
        add_button.pack(side="left", padx=5)

        self.faq_container = ttk.Frame(self.admin_frame)
        self.faq_container.pack(fill="both", expand=True)

        self.refresh_faq_list()

    def _extract_priority_facts(self, text: str) -> str:
        # Extract text between "PRIORITY FACTS (start):" and "(end facts)"
        match = re.search(r"PRIORITY FACTS \(start\):(.*?)\(end facts\)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "" 
    
    def refresh_faq_list(self):
        # Clear existing widgets
        for widget in self.faq_container.winfo_children():
            widget.destroy()

        # Create canvas + scrollbar for scrolling
        canvas = tk.Canvas(self.faq_container)
        scrollbar = ttk.Scrollbar(self.faq_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Fetch documents
        faq_docs = self.lang_model.list_faq_documents()

        for doc in faq_docs:
            frame = ttk.Frame(scrollable_frame, relief="solid", padding=5)
            frame.pack(fill="x", pady=5)

            question = doc.get("question", "Unknown Question")
            content = doc.get("page_content", "") 
            content = self._extract_priority_facts(content) # remove priority facts tags

            # Shorten content preview
            preview = (content[:100] + "...") if len(content) > 100 else content  

            # Label shows preview only
            lbl = ttk.Label(
                frame,
                text=f"Q: {question}\n{preview}",
                anchor="w",
                justify="left",
                wraplength=500
            )
            lbl.pack(side="left", fill="x", expand=True)

            # Buttons
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(side="right")

            open_btn = ttk.Button(btn_frame, text="Open", command=lambda c=content: self.open_content(c))
            open_btn.pack(side="top", padx=2, pady=1)

            update_btn = ttk.Button(btn_frame, text="Update", command=lambda d=doc: self.update_faq(d))
            update_btn.pack(side="top", padx=2, pady=1)

            delete_btn = ttk.Button(btn_frame, text="Delete", command=lambda d=doc: self.delete_faq(d))
            delete_btn.pack(side="top", padx=2, pady=1)

    def open_content(self, content):
        """Open a popup window to show full content"""
        popup = tk.Toplevel(self.root)
        popup.title("FAQ Content")
        popup.geometry("600x400")

        text_widget = tk.Text(popup, wrap="word")
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")  # read-only
        text_widget.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
     

    def add_faq(self):
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Add FAQ")
        popup.geometry("600x400")

        # Question label + entry
        q_lbl = ttk.Label(popup, text="Enter question:", anchor="w")
        q_lbl.pack(pady=5, padx=5, anchor="w")

        q_entry = ttk.Entry(popup, width=80)
        q_entry.pack(fill="x", padx=5, pady=5)

        # Answer label + multiline text box
        a_lbl = ttk.Label(popup, text="Enter answer:", anchor="w")
        a_lbl.pack(pady=5, padx=5, anchor="w")

        text_box = tk.Text(popup, wrap="word", height=15)
        text_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Scrollbar for the text box
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=text_box.yview)
        text_box.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Save button
        def save_faq():
            q = q_entry.get().strip()
            a = text_box.get("1.0", tk.END).strip()
            if q and a:
                msg = self.lang_model.add_faq_document(q, a)
                messagebox.showinfo("Info", msg)
                self.refresh_faq_list()
                popup.destroy()
            else:
                messagebox.showwarning("Warning", "Both question and answer are required.")

        save_btn = ttk.Button(popup, text="Save", command=save_faq)
        save_btn.pack(pady=10)


    def update_faq(self, doc):
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Update FAQ")
        popup.geometry("600x400")  

        # Label
        lbl = ttk.Label(popup, text="Enter new answer:", anchor="w")
        lbl.pack(pady=5, padx=5, anchor="w")

        # Multiline Text box
        text_box = tk.Text(popup, wrap="word", height=15)
        text_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Pre-fill with existing content if you want
        content = self._extract_priority_facts(doc.get("page_content", "")) # remove priority facts tags
        text_box.insert("1.0", content)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=text_box.yview)
        text_box.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")


        # Save button
        def save_update():
            new_answer = text_box.get("1.0", tk.END).strip()
            if new_answer:
                msg = self.lang_model.update_faq_document(new_answer, doc["id"])
                messagebox.showinfo("Info", msg)
                self.refresh_faq_list()
                popup.destroy()

        save_btn = ttk.Button(popup, text="Save", command=save_update)
        save_btn.pack(pady=10)


    def delete_faq(self, doc):
        msg = self.lang_model.delete_faq_document(doc["id"])
        messagebox.showinfo("Info", msg)
        self.refresh_faq_list()



