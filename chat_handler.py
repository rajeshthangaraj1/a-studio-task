import os
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain_community.vectorstores import FAISS

class ChatHandler:
    def __init__(self,vector_db_path,api_token,grok_api_token,logger):
        self.logger = logger
        self.logger.info("Initializing ChatHandler...")
        self.vector_db_path = vector_db_path
        self.groq_client = Groq(api_key=grok_api_token)
        # Initialize the embedding model using Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": api_token},
        )

    def _query_groq_model(self, prompt):
        """
        Query Groq's Llama model using the SDK.
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Ensure the model name is correct
            )
            # Return the assistant's response
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error querying Groq API: {e}")
            return f"Error querying Groq API: {e}"

    def answer_question(self, question):
        responses = []
        for root, dirs, files in os.walk(self.vector_db_path):
            for dir in dirs:
                index_path = os.path.join(root, dir, "index.faiss")
                if os.path.exists(index_path):
                    vector_store = FAISS.load_local(
                        os.path.join(root, dir), self.embeddings, allow_dangerous_deserialization=True
                    )
                    response_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=10)
                    filtered_responses = [doc.page_content for doc, score in response_with_scores]
                    responses.extend(filtered_responses)

        if responses:
            prompt = self._generate_prompt(question, responses)
            response = self._query_groq_model(prompt)
            return response


        return "No relevant documents found or context is insufficient to answer your question."

    def _generate_prompt(self, question, documents):
        """
        Generate a structured prompt that ensures responses are strictly based on
        the retrieved documents from the FAISS vector store.
        """
        # Combine the top retrieved documents into the context
        context = "\n".join(
            [f"Document {i + 1}:\n{doc.strip()}" for i, doc in enumerate(documents[:5])]
        )

        prompt = f"""
    You are an AI assistant with access to a structured knowledge base. Your role is to 
    answer user questions **strictly based on the provided documents** stored in the 
    database. You must **only use information from the retrieved documents** and 
    **do not generate answers based on assumptions or external sources**.

    ### **Knowledge Base Extracts:**
    The following documents contain relevant details:
    {context}

    ### **User Question:**
    {question}

    ### **Instructions:**
    1. **Search for Relevant Information:**
       - Extract and review the most relevant content from the retrieved documents.
       - If multiple sources exist, prioritize the most specific and recent details.

    2. **Generate a Concise and Accurate Answer:**
       - Respond in a **clear and structured** manner.
       - Use bullet points or short paragraphs for easy readability.

    3. **Avoid External or Assumptive Information:**
       - Do not provide answers beyond what is found in the knowledge base.
       - If the question cannot be answered with the available documents, state:
         - *"I couldn't find relevant information in the provided knowledge base."*

    4. **Example Response Format:**
       - **Product Details:** Provide product name, price, features, and availability.
       - **Company Policy:** Explain relevant policy details clearly.
       - **General Answer:** Direct response to the user's question.

    Your response must be **precise, fact-based, and derived solely from the provided documents.**
    """

        return prompt
