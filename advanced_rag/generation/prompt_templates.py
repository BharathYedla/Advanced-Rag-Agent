"""
Prompt Templates for Advanced RAG System.
"""

class PromptTemplates:
    """Collection of prompt templates for RAG generation."""
    
    QA_TEMPLATE = "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    @staticmethod
    def create_qa_prompt(context: str, question: str) -> str:
        return PromptTemplates.QA_TEMPLATE.format(context=context, question=question)
