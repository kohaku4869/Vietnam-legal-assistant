import google.generativeai as genai

class GoogleLLMPipeline:
    def __init__(self, api_key: str, model_name: str, **kwargs):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_kwargs = kwargs

    def invoke(self, prompt: str) -> str:
        response = self.model.generate_content(prompt, **self.model_kwargs)
        return response.text
    def stream(self, prompt: str):
        response_stream = self.model.generate_content(prompt, **self.model_kwargs, stream=True)
        for chunk in response_stream:
            # Bỏ qua các chunk rỗng có thể xuất hiện
            if chunk.text:
                yield chunk.text
    def __call__(self, input) -> str:
        if hasattr(input, "to_string"):
            prompt = input.to_string()
        elif isinstance(input, str):
            prompt = input
        elif isinstance(input, dict):
            context = input.get("context", "")
            question = input.get("question", "")
            prompt = f"{context}\n\nQuestion: {question}"
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
        return self.invoke(prompt)