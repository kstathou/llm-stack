from dataclasses import dataclass

import pandas as pd
import wandb


@dataclass
class WandbTypes:
    """Types for wandb experiments."""

    raw_data_job: str = "raw_data"
    process_data_job: str = "process_data"
    train_model_job: str = "model"
    evaluate_model_job: str = "evaluate_model"

    dataset_artifact: str = "dataset"
    model_artifact: str = "model"


class WandbRun:
    """Create a wandb session."""

    def __init__(self, project: str, job_type: str, **kwargs) -> None:
        """Create a wandb session.

        Parameters
        ----------
        project : str
            Project name on wandb.

        job_type : str
            The type of job which is being run, which is used to organize
            and differentiate steps in the ML pipeline and distinguish
            which steps created which artifacts.

        **kwargs
            Additional keyword arguments to pass to `wandb.init`.
            See https://docs.wandb.ai/ref/python/init for details.

        """
        # Use a running session or create a new one
        if wandb.run:
            self.run = wandb.run
            self.job_type = wandb.run.job_type
            self.project = wandb.run.project
        else:
            self.run = wandb.init(project=project, job_type=job_type, **kwargs)
            self.job_type = job_type
            self.project = project

    @property
    def name(self) -> str:
        """Return the run name."""
        return self.run.name

    @property
    def id(self) -> str:
        """Return the run ID."""
        return self.run.id


class ArtifactHandler(WandbRun):
    """Read and write artifacts stored in wandb."""

    def __init__(self, project: str, job_type: str, **kwargs) -> None:
        super().__init__(project, job_type, **kwargs)

    def write_artifact(
        self,
        obj: object,
        local_path: str,
        name: str,
        artifact_type: str,
        **kwargs,
    ) -> None:
        """Log an artifact in wandb. Requires a wandb session to work.

        Parameters
        ----------
        obj
            The object you want to store and log in wandb.

        local_path
            Where the object is stored locally.

        name
            A human-readable name for this artifact, which is how you
            can identify this artifact in the UI or reference it in
            use_artifact calls. The name must be unique across a project.

        artifact_type
            The type of artifact you are logging.
            Options are: 'dataset', 'model', 'metric'

        **kwargs
            Additional keyword arguments to pass to `wandb.Artifact`.
            See https://docs.wandb.ai/ref/python/artifact

        """

        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(local_path)
        else:
            raise NotImplementedError(f"Only pandas DataFrames are supported for now, not {type(obj)}")

        self._log_artifact(name=name, local_path=local_path, artifact_type=artifact_type, **kwargs)

    def _log_artifact(
        self,
        name: str,
        local_path: str,
        artifact_type: str,
        **kwargs,
    ) -> None:
        # Create the artifact
        artifact = wandb.Artifact(name=name, type=artifact_type, **kwargs)

        # Add a file
        artifact.add_file(local_path=local_path)

        self.run.log_artifact(artifact)

    def read_artifact(
        self,
        name: str,
        artifact_type: str,
        version: str = "latest",
    ) -> object:
        """Read a data or ML model artifact.

        For data artifacts, it returns a pandas dataframe. For model artifacts, it returns a
        path to the directory containing the model.

        TODO: Return a huggingface dataset instead of a pandas dataframe.

        Notes
        -----
        - Assumes that data artifacts are always stored as parquet files.

        Parameters
        ----------
        name
            The name of the artifact to download. It must contain its version
            (or `latest`) too.

        artifact_type
            Describes the artifact like `model` or `dataset`. It is used
            in the `download_path`.

        version
            Determines the version of the artifact that will be downloaded.

        """
        file_path = self._download_artifact(
            name=name,
            version=version,
        )

        if artifact_type == WandbTypes.dataset_artifact:
            return pd.read_parquet(file_path)
        else:
            raise NotImplementedError(f"Only datasets are supported for now, not {artifact_type}")

    def _download_artifact(
        self,
        name: str,
        version: str = "latest",
    ) -> str:
        artifact = self.run.use_artifact(f"{name}:{version}")

        # Download locally
        file = artifact.download()

        return file
