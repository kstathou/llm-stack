import logging

from typing import Optional
from typing import Union

import httpx

from openai import APIConnectionError
from openai import APIError
from openai import APIStatusError
from openai import AsyncOpenAI
from openai import RateLimitError
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class OpenAILLM:
    """OpenAI Wrapper for the Chat Completion API and the Embeddings.

    Handles asynchronous requests to the OpenAI Chat Completion and the Embeddings API, including:
    - Retrying failed requests
    - Handling rate limits, timeouts, exceptions

    TODO: Splitting the Chat Completion and Embeddings API into two classes would be more elegant
    but ain't nobody got time for that.

    """

    def __init__(self, api_key: str, timeout: float = 30.0, max_retries: int = 0, **kwargs) -> None:
        """Instantiate the Async OpenAI client.

        Parameters
        ----------
        api_key
            OpenAI API key.

        timeout
            Timeout for the request in seconds.

        max_retries
            Number of times to retry the request.

        **kwargs
            Additional keyword arguments to pass to the Async OpenAI client.
            See

        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    async def generate(
        self,
        messages: list[dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        extra: Optional[dict] = None,
        **openai_kwargs,
    ) -> Union[
        dict,
        str,
    ]:
        """Call OpenAI's async chat completion API and await the response.

        Handles asynchronous requests to the OpenAI Chat Completion API, including:
        - Retrying failed requests
        - Handling rate limits, timeouts, exceptions
        - Truncating long messages, if needed

        Parameters
        ----------
        messages
            Messages to send to the OpenAI Chat Completion API.

        message_kwargs
            Additional arguments to pass to the message preparation function.
            This is a dict with all your f-string placeholders and their values.

        model
            Model to use for token counting and completion.

        temperature
            Temperature to use for completion.

        seed
            Seed to use for completion.

        extra
            Additional information to return with the response.

        openai_kwargs
            Additional arguments to pass to the OpenAI Chat Completion API, like a seed.

        Returns
        -------
        Response from OpenAI Chat Completion API.

        Usage
        -----
        >>> messages = [
        >>>     {"role": "system", "name": "assistant", "content": "Tell the user a joke about it's topic of choice"},
        >>>     {"role": "user", "name": "user", "content": "Giraffes"},
        >>> ]
        >>> openai_model = OpenAILLM(api_key=API_KEY)
        >>> response = await openai_model.generate(messages, model="gpt-3.5-turbo", temperature=0.0, seed=42)
        >>> print(response)
        >>> "Why don't giraffes use computers? Because their heads are always in the clouds!"

        """

        response = await self._call(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            **openai_kwargs,
        )

        response = response.choices[0].message

        if extra:
            return {"response": response.content, **extra}

        return response

    @retry(
        retry(
            reraise=True,
            stop=stop_after_attempt(8),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=(
                retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
                | retry_if_exception_type(APIStatusError)
                | retry_if_exception_type(httpx.ReadTimeout)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    )
    async def _call(
        self,
        messages: list[dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> dict:
        """Private method to create an async OpenAI Call."""
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            **kwargs,
        )  # type:ignore

    @retry(
        retry(
            reraise=True,
            stop=stop_after_attempt(8),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=(
                retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
                | retry_if_exception_type(APIStatusError)
                | retry_if_exception_type(httpx.ReadTimeout)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
    )
    async def get_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-ada-002",
    ) -> list[list[float]]:
        """Return the embeddings for a list of text strings."""
        embeddings = await self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in embeddings.data]
