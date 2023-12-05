import modal

from modal import Image
from modal import Stub


image = (
    Image.debian_slim(python_version="3.11.5")
    .apt_install("git")
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
)

# Imports shared across functions
with image.run_inside():
    import logging

    from llm_stack.build_dataset import ArxivAPI
    from llm_stack.wandb_utils import ArtifactHandler
    from llm_stack.wandb_utils import WandbTypes

stub = Stub(name="build-ner-dataset-with-openai", image=image)


@stub.function(secret=modal.Secret.from_name("wandb-secret"))
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


@stub.local_entrypoint()
def main(local_data_path: str = "arxiv_preprints.parquet") -> None:
    """Build an NER dataset using arXiv's papers and OpenAI's LLMs."""
    logging.info("Fetching arXiv data...")
    fetch_arxiv_data.remote(local_data_path=local_data_path)


# predictions = asyncio.run(
#     main(
#         data=data,
#         messages=messages,
#         model_name=MODEL_NAME,
#         seed=seed,
#         local_dir=local_dir,
#         local_file=local_file
#         )
#     )
