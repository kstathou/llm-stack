# llm-stack

This tutorial series will show you how to build an end-to-end data flywheel for Large Language Models (LLMs).

We will be working on an entity recognition task with a custom set of entity types throughout the series. We will extract entities like `dataset`, `method`, `evaluation` and `task` from arXiv papers.

## What you will learn

How to:

- Build a training set with GPT-4 or GPT-3.5
- Fine-tune an open-source LLM
- Create a set of Evals to evaluate the model.
- Collect human feedback to improve the model.
- Deploy the model to an inference endpoint.

## Software used

- [wandb](https://wandb.ai) for experiment tracking. This is where we will record all our artifacts (datasets, models, code) and metrics.
- [modal](https://modal.com/) for running jobs on the cloud.
- [huggingface](https://huggingface.co/) for all-things-LLM.
- [argilla](https://docs.argilla.io/en/latest/) for labelling our data.

## Tutorial 1 - Generating a training set with GPT-3.5

In this tutorial, we will use GPT-3.5 to generate a training set for our entity recognition task.

```python
modal run src/llm_stack/scripts/build_dataset_ner.py
```

## Contributing

Found any mistakes or want to contribute? Feel free to open a PR or an issue.
