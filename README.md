# LLM_Auto_Eval

LLMを使ってLLMを自動評価します
主にELYZA-tasks-100用です

## 使い方

### インストール

`pip install git+https://github.com/HawkClaws/LLM_Auto_Eval.git`


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


# OpenAI APIの場合
from langchain_openai import ChatOpenAI
openai_api_key = ""
eval_llm = ChatOpenAI(
    api_key=openai_api_key, model_name="gpt-4-turbo-2024-04-09"
)

# Gemini APIの場合
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "your token"
eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


## LLM_Auto_Evalをインポートして、run
from LLM_Auto_Eval.auto_eval import run
run(llm, eval_llm)

```

##