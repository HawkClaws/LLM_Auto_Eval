from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fp:
    readme = fp.read()

DESCRIPTION = "Automatically evaluate LLMs using LLMs (mainly for ELYZA-tasks-100)"

setup(
    name="LLM_Auto_Eval",
    version="0.0.1",
    author="HawkClaws",
    packages=find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    url="https://github.com/HawkClaws/LLM_Auto_Eval",
    project_urls={"Source Code": "https://github.com/HawkClaws/LLM_Auto_Eval"},
    description=DESCRIPTION,
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    install_requires=[
        "langchain",
        "tenacity ",
        "LiteLLMJson",
    ],
)
