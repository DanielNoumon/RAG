import openai
from config import Config
from prompts import RAG_PROMPT


class AzureOpenAIClient:
    def __init__(self):
        self.config = Config()
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            api_key=self.config.AZURE_OPENAI_API_KEY,
            api_version=self.config.AZURE_OPENAI_API_VERSION
        )

    def generate_response(self, prompt: str) -> str:
        """Generate response using Azure OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def generate_rag_response(self, query: str, context_docs: list[str]) -> str:
        """Generate RAG response with context"""
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = RAG_PROMPT.format(context=context, query=query)
        return self.generate_response(prompt)
