from modal import Image
from modal import Mount
from modal import Secret
from modal import Stub


image = (
    Image.debian_slim(python_version="3.11.5")
    .apt_install("git")
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
)

# Imports shared across functions
with image.run_inside():
    import asyncio
    import os

    import pandas as pd

    from tqdm import tqdm

    from llm_stack.build_dataset import ArxivAPI
    from llm_stack.openai import FunctionTemplate
    from llm_stack.openai import MessageTemplate
    from llm_stack.openai import OpenAILLM
    from llm_stack.wandb_utils import ArtifactHandler
    from llm_stack.wandb_utils import WandbTypes


stub = Stub(name="build-ner-dataset-with-openai", image=image)


@stub.function(secret=Secret.from_name("wandb-secret"))
def fetch_arxiv_data(
    local_data_path: str,
    artifact_name: str = "arxiv-preprints",
    query: str = "LLM,large language models,gpt",
    start_date: int = 20230101,
    end_date: int = 20231205,
    max_results: int = 2000,
) -> None:
    """Fetch arxiv data from the arxiv API."""
    handler = ArtifactHandler(project="llm-stack", job_type=WandbTypes.raw_data_job)

    # Grab LLM-related papers from 2023
    preprints = ArxivAPI.search_and_parse_papers(
        query=query,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )

    # Save the raw data
    handler.write_artifact(
        obj=preprints,
        local_path=local_data_path,
        name=artifact_name,
        artifact_type=WandbTypes.dataset_artifact,
        metadata={"query": query, "start_date": start_date, "end_date": end_date},
    )

    handler.run.finish()


@stub.function(
    secrets=[Secret.from_name("wandb-secret"), Secret.from_name("openai-secret")],
    mounts=[Mount.from_local_dir("src/llm_stack/build_dataset/prompts", remote_path="/root/prompts")],
)
async def annotate_dataset_with_open_ai(
    local_data_path: str,
    raw_data_artifact: str = "arxiv-preprints",
    annotated_artifact_name: str = "preprints-with-openai-ner",
    model_name: str = "gpt-3.5-turbo-1106",
    system_message_file: str = "openai_system.json",
    user_message_file: str = "openai_ner.json",
    function_file: str = "ner_function.json",
    timeout: int = 120,
) -> None:
    """Run NER with OpenAI's LLMs."""
    handler = ArtifactHandler(project="llm-stack", job_type=WandbTypes.inference_job)

    preprints = handler.read_artifact(
        name=raw_data_artifact,
        artifact_type=WandbTypes.dataset_artifact,
    )

    # Load all prompt templates
    system_message = MessageTemplate.load(f"/root/prompts/{system_message_file}")
    user_message = MessageTemplate.load(f"/root/prompts/{user_message_file}")
    function = FunctionTemplate.load(f"/root/prompts/{function_file}")

    openai_llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)

    # Run the async tasks
    tasks = []
    for tup in preprints.itertuples():
        messages = [
            system_message.to_prompt(),
            user_message.to_prompt(text=tup.abstract),
        ]
        tasks.append(
            openai_llm.generate(
                messages=messages,
                model=model_name,
                tools=[function.to_prompt()],
                tool_choice={
                    "type": "function",
                    "function": {"name": function.name},
                },
                extra={"id": tup.arxiv_url},
            )
        )

    predictions = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await future
        if result:
            predictions.append(result)

    # Save the predictions to wandb
    arxiv_urls = []
    entities = []
    for d in predictions:
        arxiv_urls.append(d["id"])
        d.pop("id")
        entities.append(d)

    df = pd.DataFrame({"arxiv_url": arxiv_urls, "entities": entities})

    cols = ["arxiv_url", "abstract", "entities"]
    preprints = preprints.merge(df, on="arxiv_url")[cols]

    handler.write_artifact(
        obj=preprints,
        local_path=local_data_path,
        name=annotated_artifact_name,
        artifact_type=WandbTypes.dataset_artifact,
    )


@stub.local_entrypoint()
async def main(
    local_data_path_raw: str = "arxiv_preprints.parquet",
    local_data_path_ner_openai: str = "preprints_openai_ner.parquet",
) -> None:
    """Build an NER dataset using arXiv's papers and OpenAI's LLMs."""
    # Fetching the arXiv data
    fetch_arxiv_data.remote(local_data_path=local_data_path_raw)

    # NER with OpenAI
    annotate_dataset_with_open_ai.remote(local_data_path=local_data_path_ner_openai)
