from __future__ import annotations
import base64
import os
import typing
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
import openai
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import ChatCompletionUserMessageParam
from tenacity import Retrying
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from aoptk.chemical import Chemical
from aoptk.effect import Effect
from aoptk.find_chemical import FindChemical
from aoptk.literature.convert_image import ConvertImage
from aoptk.literature.convert_pdf_scan import ConvertPDFScan
from aoptk.literature.find_relevant_publication import FindRelevantPublication
from aoptk.normalization.normalize_chemical import NormalizeChemical
from aoptk.relationships.find_relationship import FindRelationship
from aoptk.relationships.relationship import Relationship
from aoptk.relationships.relationship_type import Activation
from aoptk.relationships.relationship_type import Alleviation
from aoptk.relationships.relationship_type import Causation
from aoptk.relationships.relationship_type import Induction
from aoptk.relationships.relationship_type import Inhibition
from aoptk.relationships.relationship_type import Mitigation
from aoptk.relationships.relationship_type import Prevention
from aoptk.relationships.relationship_type import Promotion
from aoptk.relationships.relationship_type import Regulation
from aoptk.relationships.relationship_type import RelationshipType

if TYPE_CHECKING:
    from collections.abc import Callable
    from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
    from tenacity import RetryCallState

# Models used when the caller does not choose one. The e-infra endpoint retires models
# periodically, which used to mean grepping tests and examples for the retired name; both now
# refer to these constants, so a retirement is a one-line change here.
#
# `DEFAULT_VISION_MODEL` is what the image and scanned-PDF tasks use. There is no vision
# capability check in the library - the supported set is whatever the four `@pytest.mark.openai`
# vision tests accept, so verify those before changing this. Vision prompts are model-sensitive,
# and the endpoint reports a retired model as a bad-request error that the test suite classifies
# as an unavailable service and xfails; a retired value therefore silences those tests rather than
# failing them. `tests/test_llm_model_catalog.py` compares these two values against
# `models/llm_models.json` for that reason, and flags it in its output when either is archived.
DEFAULT_MODEL = "gpt-oss-120b"
# The model this named was reported archived by the provider, last seen 2026-08-26; the closest
# online model of the same family took its place. Vision capability of the replacement is not yet
# confirmed by a passing vision test.
DEFAULT_VISION_MODEL = "qwen3.5-int4"

topics = {
    Inhibition(),
    Causation(),
    Activation(),
    Promotion(),
    Prevention(),
    Induction(),
    Alleviation(),
    Mitigation(),
    Regulation(),
}


def other_topics_labels(relationship_type: RelationshipType) -> str:
    """Render the labels of all topics except the given one, in stable alphabetical order.

    Sorting is required for reproducibility: set iteration order depends on hashing and
    varies between processes, which would make rendered prompts (and thus LLM behavior)
    nondeterministic across runs.

    Args:
        relationship_type (RelationshipType): The relationship type to exclude.
    """
    remaining = {topic for topic in topics if topic != relationship_type}
    return ", ".join(sorted(topic.positive for topic in remaining))


class LLMFailureError(Exception):
    """Base class for LLM failures that leave aoptk without a usable answer.

    The message says what went wrong. The no-argument `__init__` this class used to define made
    that impossible: an empty completion and a blocked model both surfaced as the bare class
    name, and the completion that carried the reason was discarded before construction.
    """


class LLMRequestError(LLMFailureError):
    """Raised when the request itself failed, at the transport or at the endpoint.

    Wraps the openai SDK's `APIError` family - HTTP status errors, connection errors, timeouts
    and response-validation failures alike - so one `except LLMFailureError` covers aoptk's
    failure domain without the caller importing SDK types. The SDK exception is chained with
    `raise ... from`, which keeps its status code reachable for inspection and keeps the test
    suite's service-outage classifier working, since that walks the cause chain.

    Retrying is deliberately left to `_prompt`, which treats this like any other failure. That
    multiplies with the SDK's own retries: with `prompt_attempts=3` and `max_retries=3` one
    hopeless call costs up to 9 requests, so a permanent failure - bad credentials, a blocked
    model - is reported only after all of them. That cost is the price of one hierarchy.
    """


class InvalidResponseError(Exception):
    """Raised when the LLM answers with content rejected by a task-supplied validator.

    Carries the rejected answer so the caller can still use it once the retry budget is
    exhausted. Deliberately not a subclass of LLMFailureError: exhausting the budget on
    rejected content still yields a (best effort) answer, unlike exhausting it on an empty one.
    """

    def __init__(self, response: str):
        super().__init__(f"LLM response rejected by validator: {response!r}")
        self.response = response


class TextGenerationAPI(
    FindChemical,
    FindRelationship,
    NormalizeChemical,
    ConvertPDFScan,
    ConvertImage,
    FindRelevantPublication,
):
    """Text generation API using OpenAI."""

    role: Literal["user"] = "user"
    temperature: float = 0
    top_p: float = 1
    load_dotenv()
    client: OpenAI
    max_retries: int = 3
    timeout: int = 120
    # Retrying of the answer, on top of the transport-level retrying of `max_retries`:
    # a request that succeeds technically but yields unusable content (empty, or rejected
    # by a task-supplied validator) is re-requested up to `prompt_attempts` times. The two
    # layers multiply - with `max_retries=3` inside `prompt_attempts=3` a single call makes
    # at most 3 x 3 = 9 requests - so keep both small.
    prompt_attempts: int = 3
    prompt_backoff_multiplier: float = 0.5
    prompt_backoff_max: float = 30
    prompts_dir: Path = Path(__file__).resolve().parent / "prompts"
    chemical_prompt_template: str = "chemical_prompt.txt"
    relationship_text_prompt_template: str = "relationship_text_prompt.txt"
    relationship_text_images_prompt_template: str = "relationship_text_images_prompt.txt"
    relationships_table_prompt_template: str = "relationships_table_prompt.txt"
    normalization_prompt_template: str = "normalization_prompt.txt"
    convert_pdf_scan_prompt_template: str = "convert_pdf_scan_prompt.txt"
    convert_image_prompt_template: str = "convert_image_prompt.txt"
    find_relevant_publications_prompt_template: str = "find_relevant_publications_prompt.txt"
    categorize_text_prompt_template: str = "categorize_text_prompt.txt"

    specification_relationship_text_prompt: str = ""
    specification_categorization_prompt: str = ""

    invalid_chemical_response_patterns: typing.ClassVar[list[str]] = ["\n-", "\n*", "\n1.", "\n"]
    invalid_normalization_response_patterns: typing.ClassVar[list[str]] = ["\n-", "\n*", "\n1.", "\n", " ; "]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = "https://llm.ai.e-infra.cz/v1",
        api_key: str | None = os.environ.get("CERIT_API_KEY"),
    ):
        self.model = model
        self.url = url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.url,
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def find_relationships_in_text(
        self,
        text: str,
        chemicals: list[Chemical],
        effects: list[Effect],
        relationship_types: list[RelationshipType],
    ) -> list[Relationship]:
        """Find relationships between chemicals and effects.

        Args:
            text (str): The input text.
            chemicals (list[Chemical]): List of chemical entities.
            effects (list[Effect]): List of effect entities.
            relationship_types (list[RelationshipType]): The relationship types to classify.
        """
        relationships = []
        for chemical, effect, relationship_type in product(chemicals, effects, relationship_types):
            if (response := self._relationship_prompt(text, chemical, effect, relationship_type)) and (
                relationship := self._select_relationship_type(response, relationship_type)
            ):
                relationships.append(
                    Relationship(relationship_type=relationship, chemical=chemical, effect=effect, context=text),
                )
        return relationships

    def _relationship_prompt(
        self,
        text: str,
        chemical: Chemical,
        effect: Effect,
        relationship_type: RelationshipType,
    ) -> str:
        """Classify the relationship between a chemical and an effect.

        Args:
            text (str): The input text.
            chemical (Chemical): The chemical entity.
            effect (Effect): The effect entity.
            relationship_type (RelationshipType): The relationship type to classify.
        """
        content = self._render_prompt(
            self.relationship_text_prompt_template,
            text=text,
            chem=chemical.name,
            effect=effect.name,
            rel_type=relationship_type,
            other_topics=other_topics_labels(relationship_type),
            specification_relationship_text_prompt=self.specification_relationship_text_prompt,
        )

        return self._prompt(content)

    def _render_prompt(self, template_name: str, **context: object) -> str:
        template_path = self.prompts_dir / template_name
        with template_path.open(encoding="utf-8") as template_file:
            template_content = template_file.read()
        return str(Template(template_content).render(**context))

    def _prompt(
        self,
        content: str | list[ChatCompletionContentPartParam],
        validator: Callable[[str], bool] | None = None,
    ) -> str:
        """Request a completion, retrying answers the transport accepts but we cannot use.

        A completion counts as unusable when it is empty (`LLMFailureError`) or when
        `validator` rejects it (`InvalidResponseError`). Both are retried with jittered
        backoff. Once the budget is exhausted an empty answer keeps raising
        `LLMFailureError`, while a merely rejected answer is returned as a best effort - the
        caller's own validation still applies to it, so no task starts trusting it.

        Args:
            content (str | list[ChatCompletionContentPartParam]): Prompt text or content parts.
            validator (Callable[[str], bool] | None): Rejects an answer worth re-requesting.
        """
        retrying = Retrying(
            retry=retry_if_exception_type((LLMFailureError, InvalidResponseError)),
            wait=wait_random_exponential(multiplier=self.prompt_backoff_multiplier, max=self.prompt_backoff_max),
            stop=stop_after_attempt(self.prompt_attempts),
            retry_error_callback=self._exhausted,
        )
        return retrying(self._single_prompt, content, validator)

    @staticmethod
    def _exhausted(retry_state: RetryCallState) -> str:
        """Return the best effort answer after the retry budget ran out, or re-raise."""
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, InvalidResponseError):
            return exception.response
        if exception is not None:
            raise exception
        # tenacity only calls this once attempts are exhausted, and an exhausted run always has an
        # outcome; getting here means the retry layer itself misbehaved, so say that rather than
        # invent a cause.
        message = f"retried {retry_state.attempt_number} time(s) and no failure was reported"
        raise LLMFailureError(message)

    def _single_prompt(
        self,
        content: str | list[ChatCompletionContentPartParam],
        validator: Callable[[str], bool] | None,
    ) -> str:
        """Perform one completion request, rejecting empty or invalid answers."""
        messages: list[ChatCompletionUserMessageParam] = [
            {
                "role": self.role,
                "content": content,
            },
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=messages,
            )
        except openai.APIError as error:
            raise LLMRequestError(self._request_failure_message(error)) from error

        choice = completion.choices[0]
        if not (response := choice.message.content):
            raise LLMFailureError(self._empty_completion_message(choice))
        response = response.strip()
        if not response:
            raise LLMFailureError(self._empty_completion_message(choice, whitespace=True))
        if validator is not None and not validator(response):
            raise InvalidResponseError(response)
        return response

    def _request_failure_message(self, error: openai.APIError) -> str:
        """Describe a failed request without repeating anything secret.

        The SDK's own message is kept because it carries the status text the endpoint returned,
        which is the part that explains the failure. Nothing from the client is added to it.
        """
        status = getattr(error, "status_code", None)
        where = f"status {status}" if isinstance(status, int) else type(error).__name__
        return f"LLM request to {self.url!r} with model {self.model!r} failed ({where}): {error}"

    def _empty_completion_message(self, choice: ChatCompletionChoice, whitespace: bool = False) -> str:
        """Describe a completion that arrived with nothing usable in it.

        `finish_reason` and `refusal` are the two fields that distinguish the cases this covers -
        a truncated generation, a content-policy refusal, a model that simply had nothing to say -
        and the endpoint reports no other reason for an empty answer. Neither is guaranteed to be
        present, so both are read defensively. The API key is never part of any of this.

        Args:
            choice: The choice whose content was empty.
            whitespace: Whether the content was whitespace rather than absent.
        """
        what = "only whitespace" if whitespace else "no content"
        finish_reason = getattr(choice, "finish_reason", None) or "unknown"
        refusal = getattr(choice.message, "refusal", None)
        detail = f", refusal={refusal!r}" if refusal else ""
        return f"LLM {self.model!r} returned {what} (finish_reason={finish_reason}{detail})"

    def _select_relationship_type(self, response: str, relationship_type: RelationshipType) -> str | None:
        """Select the relationship type based on the response.

        Args:
            response (str): The response from the model indicating the relationship type.
            relationship_type (RelationshipType): The relationship type to classify.
        """
        if response == relationship_type.positive:
            return relationship_type.positive
        if response == relationship_type.negative:
            return relationship_type.negative
        return None

    def find_chemicals(self, text: str) -> list[Chemical]:
        """Find chemicals in the given text.

        A response rejected by the chemical validation rules is re-requested before it is
        parsed, and the last attempt is used if none of them is accepted.

        Args:
            text (str): The input text to search for chemicals.
        """
        validator = self._response_validator(self.invalid_chemical_response_patterns)
        response = self._prompt(self._render_prompt(self.chemical_prompt_template, text=text), validator).lower()
        if response == "none" or self.is_invalid_response(response, self.invalid_chemical_response_patterns):
            return []
        return [Chemical(name=chem.strip().lower()) for chem in response.split(" ; ")] if response.strip() else []

    def _response_validator(self, invalid_response_patterns: list[str]) -> Callable[[str], bool]:
        """Build a `_prompt` validator that rejects answers matching the given patterns.

        Args:
            invalid_response_patterns (list[str]): Patterns that mark a response as invalid.
        """
        return lambda response: not self.is_invalid_response(response, invalid_response_patterns)

    def is_invalid_response(self, response: str, invalid_response_patterns: list[str]) -> bool:
        r"""Check if the response from the model is invalid for chemical extraction.

        Args:
            response (str): The response from the model.
            invalid_response_patterns (list[str]): List of patterns that indicate an invalid response.
        """
        return bool(any(pattern in response.lower() for pattern in invalid_response_patterns))

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode the image at the given path to a base64 string and return MIME type.

        Args:
            image_path (str): The path to the image to encode.

        Returns:
            tuple[str, str]: A tuple of (base64_encoded_image, mime_type).
        """
        ext = Path(image_path).suffix.lower()
        mime_type = f"image/{ext[1:]}"

        with Path(image_path).open("rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image, mime_type

    def _process_colon_separated_relationships_response(
        self,
        response: str,
        effect: Effect,
        relationship_type: RelationshipType,
        image_path: str,
    ) -> list[Relationship]:
        """Process the response from the model that is colon seperated.

        Args:
            response (str): The response from the model.
            effect (Effect): The effect entity.
            relationship_type (RelationshipType): The relationship type to classify.
            context (str): The path to the image, used for context in the relationship.
            image_path (str): The path to the image, used for context in the relationship.
        """
        relationships = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if " : " not in line:
                continue

            chem_name, classification = line.split(" : ", 1)
            chem_name = chem_name.strip().lower()
            classification = classification.strip().lower()

            relationship = self._select_relationship_type(classification, relationship_type)
            if relationship is None:
                continue

            relationships.append(
                Relationship(
                    relationship_type=relationship,
                    chemical=Chemical(name=chem_name),
                    effect=effect,
                    context=Path(image_path).stem,
                ),
            )

        return relationships

    def find_relationships_in_table(
        self,
        table_df: pd.DataFrame,
        effects: list[Effect],
        relationship_types: list[RelationshipType],
    ) -> list[Relationship]:
        """Find relationships between chemicals and effects in a table.

        Args:
            table_df (pd.DataFrame): Pandas DataFrame.
            relationship_types (list[RelationshipType]): The relationship types to classify.
            effects (list[Effect]): List of effect entities.
        """
        relationships = []
        for effect, relationship_type in product(effects, relationship_types):
            relationships.extend(
                self._classify_relationships_in_table(
                    table_df,
                    effect,
                    relationship_type,
                ),
            )
        return relationships

    def _classify_relationships_in_table(
        self,
        table_df: pd.DataFrame,
        effect: Effect,
        relationship_type: RelationshipType,
    ) -> list[Relationship]:
        """Classify relationships between chemicals and an effect in a table.

        Args:
            table_df (pd.DataFrame): Pandas DataFrame.
            effect (Effect): The effect entity.
            relationship_type (RelationshipType): The relationship type to classify.

        Returns:
            list[Relationship]: List of relationships found in the table.
        """
        table_text = table_df.to_csv(index=False)

        content = self._render_prompt(
            self.relationships_table_prompt_template,
            effect=effect.name,
            rel_type=relationship_type,
            table=table_text,
        )

        if response := self._prompt(content):
            return self._process_colon_separated_relationships_response(response, effect, relationship_type, "table")
        return []

    def normalize_chemical(self, chemical: Chemical, chemical_list: list[Chemical]) -> Chemical:
        """Normalize the chemical name by finding a matching name in the chemical list.

        Args:
            chemical (Chemical): The chemical to normalize.
            chemical_list (list[Chemical]): The list of chemicals to match against.

        Returns:
            Chemical: The normalized chemical.
        """
        if matching_name := self._find_matching_name(chemical, chemical_list):
            chemical.heading = matching_name.name
        return chemical

    def _find_matching_name(self, chemical: Chemical, chemical_list: list[Chemical]) -> Chemical | None:
        """Find a matching chemical name in the chemical list.

        A response rejected by the normalization validation rules is re-requested before it
        is used, and the last attempt is used if none of them is accepted.

        Args:
            chemical (Chemical): The chemical to find a match for.
            chemical_list (list[Chemical]): The list of chemicals to match against.

        Returns:
            Chemical: The matching chemical name, or None if no match is found.
        """
        content = self._render_prompt(
            self.normalization_prompt_template,
            chem=chemical.name,
            list_of_chemical_names="\n".join([chem.name for chem in chemical_list]),
        )
        validator = self._response_validator(self.invalid_normalization_response_patterns)

        response = self._prompt(content, validator).lower()
        if response == "none" or self.is_invalid_response(response, self.invalid_normalization_response_patterns):
            return chemical
        return Chemical(name=response)

    def convert_pdf_scan(
        self,
        img_base64: str,
        mime_type: str,
    ) -> str:
        """Extract text from a base64-encoded image.

        Args:
            img_base64 (str): Base64-encoded image data.
            mime_type (str): MIME type of the image. Defaults to "image/jpeg".

        Returns:
            str: Extracted text from the image.
        """
        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "text",
                "text": self._render_prompt(self.convert_pdf_scan_prompt_template),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64.strip()}"}},
        ]

        if response := self._prompt(content):
            return response
        return ""

    def find_relationships_in_text_and_images(
        self,
        text: str,
        image_paths: list[str],
        relationship_types: list[RelationshipType],
        effects: list[Effect],
    ) -> list[Relationship]:
        """Find relationships between chemicals and effects in the given text and images combined.

        Args:
            text (str): The input text.
            image_paths (list[str]): List of paths to images.
            relationship_types (list[RelationshipType]): The relationship types to classify.
            effects (list[Effect]): List of effect entities.
        """
        relationships = []
        for effect, relationship_type in product(effects, relationship_types):
            relationships.extend(
                self._classify_relationships_in_text_and_images(text, image_paths, effect, relationship_type),
            )
        return relationships

    def _classify_relationships_in_text_and_images(
        self,
        text: str,
        image_paths: list[str],
        effect: Effect,
        relationship_type: RelationshipType,
    ) -> list[Relationship]:
        """Classify relationships between chemicals and an effect in the given text and images combined.

        Args:
            text (str): The input text.
            image_paths (list[str]): List of paths to images.
            effect (Effect): The effect entity.
            relationship_type (RelationshipType): The relationship type to classify.
        """
        encoded_images = [self._encode_image(image_path) for image_path in image_paths]

        relationships = []

        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "text",
                "text": self._render_prompt(
                    self.relationship_text_images_prompt_template,
                    text=text,
                    effect=effect.name,
                    rel_type=relationship_type,
                    other_topics=other_topics_labels(relationship_type),
                ),
            },
        ]

        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img}"},
            }
            for img, mime_type in encoded_images
        )

        response = self._prompt(content)

        if response:
            relationships.extend(
                self._process_colon_separated_relationships_response(
                    response,
                    effect,
                    relationship_type,
                    "text and images",
                ),
            )
        return relationships

    def convert_image(
        self,
        image_path: str,
        text: str,
    ) -> str:
        """Convert an image to text.

        Args:
            image_path (str): Path to the image.
            text (str): The full text of the publication for context.
        """
        base64_image, mime_type = self._encode_image(image_path)

        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "text",
                "text": self._render_prompt(self.convert_image_prompt_template, text=text),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
        ]

        if response := self._prompt(content):
            return response
        return ""

    def find_relevant_publications(self, question: str, text: str) -> bool | None:
        """Answer the question based on a given text.

        Args:
            question (str): The question to search for relevant publications.
            text (str): The extracted text of the publication.
        """
        if response := self._prompt(
            self._render_prompt(self.find_relevant_publications_prompt_template, question=question, text=text),
        ).lower():
            if response == "yes":
                return True
            if response == "no":
                return False
        return None

    def categorize_text(self, text: str, categories: list[str]) -> str | None:
        """Categorize the given text into one of the specified categories.

        Args:
            text (str): The text to categorize.
            categories (list[str]): The list of available categories.

        Returns:
            str | None: The categorized label or None if no match is found.
        """
        if response := self._prompt(
            self._render_prompt(self.categorize_text_prompt_template, text=text, categories=", ".join(categories)),
        ).lower():
            if response == "none":
                return None
            return response
        return None
