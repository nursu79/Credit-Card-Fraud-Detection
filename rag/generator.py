import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

class FraudExplanationGenerator:
    def __init__(self):
        print("⏳ Loading Vector Database and Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        
        # We need to allow dangerous deserialization as we trust the local index we just built
        self.vector_db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.google_api_key:
            print("🌐 Using Google Gemini for AI Explanations.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.google_api_key)
        elif self.openai_api_key:
            print("🌐 Using OpenAI GPT for AI Explanations.")
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=self.openai_api_key)
        else:
            self.llm = None
            print("⚠️ No API Key found in environment variables. Falling back to rule-based explanation generator.")

    def generate_explanation(self, transaction_data, fraud_probability, risk_level):
        """
        Retrieves relevant fraud patterns from FAISS and generates an explanation.
        """
        # Step 1: Create a search query based on the transaction data characteristics
        query_elements = []
        if float(transaction_data.get("Amount", 0)) > 500:
            query_elements.append("High transaction amount")
        if risk_level == "HIGH":
            query_elements.append("High velocity of transactions")
            
        search_query = " ".join(query_elements) if query_elements else "suspicious credit card transaction"
        
        # Step 2: Retrieve similar patterns from Vector DB
        docs = self.vector_db.similarity_search(search_query, k=2)
        retrieved_patterns = "\n".join([f"- {d.metadata['title']}: {d.page_content}" for d in docs])
        recommended_actions = [d.metadata['recommended_action'] for d in docs]
        
        # Step 3: Generate Explanation using LLM or Fallback Rule-Based Heuristics
        if self.llm:
            try:
                template = """
                You are a senior fraud investigator at a fintech company. 
                Analyze the following transaction data and retrieved fraud patterns to generate a concise, professional explanation of why this transaction is flagged or not.
                
                Transaction Risk Level: {risk_level}
                Fraud Probability: {fraud_probability}
                Transaction Amount: {amount}
                
                Found patterns in knowledge base:
                {patterns}
                
                Based on the above, write an 'Investigation Explanation':
                """
                prompt = PromptTemplate(
                    input_variables=["risk_level", "fraud_probability", "amount", "patterns"],
                    template=template
                )
                
                chain = prompt | self.llm | StrOutputParser()
                
                explanation = chain.invoke({
                    "risk_level": risk_level,
                    "fraud_probability": round(fraud_probability, 3),
                    "amount": transaction_data.get("Amount", 0),
                    "patterns": retrieved_patterns
                }).strip()
            except Exception as e:
                print(f"⚠️ OpenAI Error ({e}). Falling back to local explainer.")
                # Local fallback logic duplicated here for safety
                if risk_level == "LOW":
                    explanation = "The transaction aligns with normal spending behavior. No significant risk indicators found."
                else:
                    titles = [d.metadata['title'] for d in docs]
                    explanation = f"The transaction shows high risk indicators. The model predicted anomalous behavior ({round(fraud_probability * 100, 1)}% confidence). It matches known patterns such as: {', '.join(titles)}."
        else:
            # Fallback when no OpenAI API Key is provided
            if risk_level == "LOW":
                explanation = "The transaction aligns with normal spending behavior. No significant risk indicators found."
            else:
                titles = [d.metadata['title'] for d in docs]
                explanation = f"The transaction shows high risk indicators. The model predicted anomalous behavior ({round(fraud_probability * 100, 1)}% confidence). It matches known patterns such as: {', '.join(titles)}."
        
        return {
            "fraud_probability": float(round(fraud_probability, 3)),
            "risk_level": risk_level,
            "fraud_patterns": [d.metadata['title'] for d in docs],
            "ai_explanation": explanation,
            "recommended_action": recommended_actions[0] if recommended_actions else "Monitor transaction."
        }
