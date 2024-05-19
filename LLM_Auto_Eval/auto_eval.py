import csv
from datetime import datetime
from typing import Protocol
from langchain_core.language_models.chat_models import BaseChatModel
from .load_elyza_task import ElyzaTasks100
from .evaluation import evaluation

tasks = ElyzaTasks100()


class LLM(Protocol):
    def __call__(self, input_str: str) -> str: ...


def run(llm: LLM, eval_llm: BaseChatModel):
    now = datetime.now().strftime("%Y%m%d%H%M%S").zfill(14)
    output_file = f"result_test_{now}.csv"
    fieldnames, row_datas = tasks.get_test_data()

    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = fieldnames + ["result_output"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_datas:
            input_text = row["input"]
            result_output = llm(input_text)

            row["result_output"] = result_output
            writer.writerow(row)
    evaluation(output_file, eval_llm)
