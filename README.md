# LLM_Auto_Eval

LLMを使ってLLMを自動評価します
主にELYZA-tasks-100用です

## 使い方

### git クローンと必要なものインストール

`git clone https://github.com/HawkClaws/LLM_Auto_Eval.git`

`pip install langchain langchain_openai langchain_google_genai tenacity LiteLLMJson`


### トークン設定

`evaluation.py`で、OpenAIあるいはGeminiのトークンを設定してください！  
LangChainのBaseChatModelを使っているので、対応しているLLMであれば何でもOKです

### 実行する
して、`def __call__(self, input_str: str) -> str:`の関数を作ってrunするだけ！

```python
from transformers import AutoTokenizer
import transformers
import torch

model = "HawkClaws/multi_vecteus_7B_JP"
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

## `def __call__(self, input_str: str) -> str:`の関数を作る
def llm(prompt):
    messages = [{"role": "user", "content": prompt}]

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    generated_text = generated_textoutputs[0]["generated_text"]
    return generated_text

## LLM_Auto_Evalをインポートして、run
from LLM_Auto_Eval.auto_eval import run
run(llm)
```

##