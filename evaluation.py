import csv
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_fixed
from lite_llm_json import LiteLLMJson

MODEL: BaseChatModel = None

# OpenAI APIの場合
# openai_api_key = ""
# MODEL = ChatOpenAI(
#     api_key=openai_api_key, model_name="gpt-4-turbo-2024-04-09"
# )

# Gemini APIの場合
os.environ["GOOGLE_API_KEY"] = "your token"
MODEL = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


@retry(wait=wait_fixed(30), stop=stop_after_attempt(3))
def eval_by_llm(model: BaseChatModel, pred, input_text, output_text, eval_aspect):
    prompt = f"""あなたは採点者です。

    問題, 正解例, 採点基準, 回答 が与えられます。

    採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

    # 問題

    {input_text}

    # 正解例

    {output_text}

    # 採点基準

    基本的な採点基準

    - 1点: 誤っている、 指示に従えていない
    - 2点: 誤っているが、方向性は合っている
    - 3点: 部分的に誤っている、 部分的に合っている
    - 4点: 合っている
    - 5点: 役に立つ

    基本的な減点項目

    - 不自然な日本語: -1点
    - 部分的に事実と異なる内容を述べている: -1点
    - 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

    問題固有の採点基準

    {eval_aspect}

    # 回答

    {pred}
    """
    llm_json = LiteLLMJson(SCORE_SCHEMA)
    generated_prompt = llm_json.generate_prompt(prompt)
    output_text = model.invoke(generated_prompt).content
    json_data = llm_json.parse_response(output_text)
    return int(json_data["score"])


def evaluation(input_file):

    output_file = os.path.join(
        os.path.dirname(input_file),
        "evaluated_"
        + os.path.basename(os.path.splitext(input_file)[0])
        + os.path.splitext(input_file)[1],
    )

    with open(input_file, "r", encoding="utf-8") as csvfile, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ["score"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            input_text = row["input"]
            output_text = row["output"]
            eval_aspect = row["eval_aspect"]
            pred = row["result_output"]
            print("=============pred=============")
            print(pred)
            score = eval_by_llm(MODEL, pred, input_text, output_text, eval_aspect)
            print("=============score=============")
            print(score)
            row["score"] = score

            writer.writerow(row)
