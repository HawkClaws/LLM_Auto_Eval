import csv
import os
import requests

# ファイルパスとURLを指定
FILE_PATH = "./test.csv"
FILE_URL = "https://huggingface.co/datasets/elyza/ELYZA-tasks-100/resolve/main/test.csv?download=true"


class ElyzaTasks100:
    def get_test_data(self, file_path=None) -> tuple[list[str], list[dict]]:
        if file_path == None:
            file_path = FILE_PATH
        if not os.path.exists(file_path):
            self.download_file(FILE_URL, file_path)
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            list_of_dicts = [row for row in reader]
        return list(reader.fieldnames), list_of_dicts

    # ファイルが存在しない場合はダウンロードする関数
    def download_file(self, url, file_path):
        response = requests.get(url)
        with open(file_path, "wb") as file:
            file.write(response.content)

if __name__ == '__main__':
    et = ElyzaTasks100()
    _,datas = et.get_test_data()
    print(datas)
