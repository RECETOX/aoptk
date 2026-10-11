"""Retry behaviour of `TextGenerationAPI._prompt`, exercised against a mocked endpoint.

Everything here is offline by design: `CERIT_API_KEY` is faked and the completion endpoint is
replaced, so what is under test is aoptk's own handling - retries, and how a failure is turned
into an exception. What the real gateway returns on a bad day is a separate question, answered by
the live `@pytest.mark.openai` tests.
"""

# ruff: noqa: PLR2004
# ruff: noqa: SLF001
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import openai
import pytest
from pytest_mock import MockerFixture
from aoptk.chemical import Chemical
from aoptk.text_generation_api import LLMFailureError
from aoptk.text_generation_api import LLMRequestError
from aoptk.text_generation_api import TextGenerationAPI
from tests.service_availability import is_server_problem

CHEMICAL_TEXT = "Paracetamol is toxic to the liver."
# Not the real thing, and not empty: a recognisable string is what makes the test asserting that
# no message leaks the credential mean something.
MOCK_API_KEY = "mock-api-key"
# A bulleted answer only counts as invalid once it carries a newline, because the patterns are
# matched against the answer after stripping - a bullet at the very start is stripped away.
BULLETED_RESPONSE = "- thioacetamide\n- methotrexate"
TWO_CHEMICALS_RESPONSE = "thioacetamide ; methotrexate"
TRICHLORFON = Chemical(name="trichlorfon")
CHEMICAL_LIST = [Chemical(name="thioacetamide")]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the retry backoff so tests that exhaust the attempt budget stay fast."""
    monkeypatch.setattr(TextGenerationAPI, "prompt_backoff_multiplier", 0.0)


def completion(
    content: str | None,
    finish_reason: str | None = "stop",
    refusal: str | None = None,
) -> SimpleNamespace:
    """Build the minimal stand-in for a completion, holding just what the code reads.

    `finish_reason` and `refusal` default to a normal successful stop, because only the tests
    about failure messages care about them.
    """
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def endless(*contents: str | None | SimpleNamespace) -> Iterator[SimpleNamespace]:
    """Yield one completion per answer, then repeat the last one forever.

    An answer is normally just the message content; a completion built by `completion` is also
    accepted, for the tests that need a specific `finish_reason` or `refusal`.

    Repeating rather than raising keeps tests that assert on the attempt budget honest: a
    surplus call returns an answer instead of blowing up with StopIteration.
    """
    answers = [content if isinstance(content, SimpleNamespace) else completion(content) for content in contents]
    yield from answers
    while answers:
        yield answers[-1]


def mocked_api(mocker: MockerFixture, *contents: str | None | SimpleNamespace) -> tuple[TextGenerationAPI, Mock]:
    """Build an API whose completion endpoint answers with `contents`, one per call.

    Only the endpoint facing the service is mocked, so the empty-answer and validator logic
    the retry layer reacts to is the real one. The dummy key keeps these tests independent of
    `CERIT_API_KEY`: nothing here reaches the service, so they must not be skipped without it.
    Tests that need the endpoint to raise pass any content and then replace the `side_effect`.

    Returns:
        tuple[TextGenerationAPI, Mock]: The API, and the mock counting completion calls.
    """
    return failing_api(mocker, endless(*contents))


def failing_api(mocker: MockerFixture, side_effect: object) -> tuple[TextGenerationAPI, Mock]:
    """Build an API whose completion endpoint fails with `side_effect` on every call.

    Exists alongside `mocked_api` because the endpoint is reached through the SDK's own typed
    attribute: reassigning `side_effect` on the returned object is something mypy rightly
    rejects, so the failure has to be installed at patch time.
    """
    api = TextGenerationAPI(api_key=MOCK_API_KEY)
    create = mocker.patch.object(api.client.chat.completions, "create", side_effect=side_effect)
    return api, create


def test_prompt_retries_empty_answer_and_returns_the_first_usable_one(mocker: MockerFixture):
    """Two answers the transport accepted but left empty are re-requested, the third is used."""
    api, create = mocked_api(mocker, None, "", "PCB-123")

    assert api._prompt("some prompt") == "PCB-123"
    assert create.call_count == 3


def test_prompt_raises_llm_failure_once_the_attempt_budget_is_exhausted(mocker: MockerFixture):
    """An answer that stays empty fails loudly instead of collapsing into an empty result."""
    api, create = mocked_api(mocker, "")

    with pytest.raises(LLMFailureError):
        api._prompt("some prompt")
    assert create.call_count == TextGenerationAPI.prompt_attempts


def test_extraction_fails_loudly_when_an_attempt_is_empty_after_an_invalid_one(mocker: MockerFixture):
    """A rejected answer is not passed off as an answer once a later attempt comes back empty."""
    api, create = mocked_api(mocker, BULLETED_RESPONSE, None)

    with pytest.raises(LLMFailureError):
        api.find_chemicals(CHEMICAL_TEXT)
    assert create.call_count == TextGenerationAPI.prompt_attempts


def test_find_chemicals_retries_invalid_answer_and_keeps_the_valid_one(mocker: MockerFixture):
    """A bulleted answer is re-requested, and the semicolon separated one is parsed."""
    api, create = mocked_api(mocker, BULLETED_RESPONSE, TWO_CHEMICALS_RESPONSE)

    result = api.find_chemicals(CHEMICAL_TEXT)

    assert [chem.name for chem in result] == ["thioacetamide", "methotrexate"]
    assert create.call_count == 2


def test_find_chemicals_returns_no_chemicals_when_every_answer_is_invalid(mocker: MockerFixture):
    """The documented empty-list contract holds even when every attempt is rejected."""
    api, create = mocked_api(mocker, BULLETED_RESPONSE)

    assert api.find_chemicals(CHEMICAL_TEXT) == []
    assert create.call_count == TextGenerationAPI.prompt_attempts


def test_normalization_retries_invalid_answer_and_keeps_the_valid_one(mocker: MockerFixture):
    """An answer naming two chemicals is re-requested, the single name is used."""
    api, create = mocked_api(mocker, TWO_CHEMICALS_RESPONSE, "thioacetamide")

    matched = api._find_matching_name(TRICHLORFON, CHEMICAL_LIST)

    assert matched is not None
    assert matched.name == "thioacetamide"
    assert create.call_count == 2


def test_normalization_keeps_the_original_name_when_every_answer_is_invalid(mocker: MockerFixture):
    """The documented fallback holds even when every attempt is rejected."""
    api, create = mocked_api(mocker, TWO_CHEMICALS_RESPONSE)

    assert api._find_matching_name(TRICHLORFON, CHEMICAL_LIST) == TRICHLORFON
    assert create.call_count == TextGenerationAPI.prompt_attempts


def status_error(
    status: int,
    body: str,
    error: type[openai.APIStatusError] = openai.APIStatusError,
) -> openai.APIStatusError:
    """Build an SDK status error as the client would raise it for a given HTTP status.

    The request it points at is a stand-in: the SDK needs one to construct the error, and the
    endpoint address in it is never what the assertions look at.
    """
    request = httpx.Request("POST", "https://llm.example.invalid/v1")
    return error(f"Error code: {status} - {body}", response=httpx.Response(status, request=request), body=None)


def test_empty_completion_reports_the_model_and_finish_reason(mocker: MockerFixture):
    """The failure names the model and says why the endpoint had nothing to return."""
    api, _ = mocked_api(mocker, None)

    with pytest.raises(LLMFailureError) as raised:
        api._prompt("some prompt")

    message = str(raised.value)
    assert api.model in message
    assert "finish_reason=stop" in message
    assert "no content" in message


def test_empty_completion_reports_the_finish_reason_of_a_truncated_answer(mocker: MockerFixture):
    """A generation cut off mid-answer is distinguishable from one that never started."""
    api, _ = mocked_api(mocker, completion(None, finish_reason="length"))

    with pytest.raises(LLMFailureError, match="finish_reason=length"):
        api._prompt("some prompt")


def test_empty_completion_reports_a_refusal(mocker: MockerFixture):
    """A content-policy refusal is surfaced rather than reported as an unexplained empty answer."""
    api, _ = mocked_api(mocker, completion(None, finish_reason="content_filter", refusal="I cannot help with that."))

    with pytest.raises(LLMFailureError) as raised:
        api._prompt("some prompt")

    assert "content_filter" in str(raised.value)
    assert "I cannot help with that." in str(raised.value)


def test_whitespace_only_completion_is_distinguished_from_an_absent_one(mocker: MockerFixture):
    """Whitespace content and no content are different endpoint behaviours, reported apart."""
    api, _ = mocked_api(mocker, "   ")

    with pytest.raises(LLMFailureError, match="only whitespace"):
        api._prompt("some prompt")


@pytest.mark.parametrize(
    ("status", "sdk_error"),
    [
        (429, openai.RateLimitError),
        (503, openai.APIStatusError),
        (401, openai.AuthenticationError),
    ],
)
def test_endpoint_errors_are_wrapped_as_llm_failures(
    mocker: MockerFixture,
    status: int,
    sdk_error: type[openai.APIStatusError],
):
    """A failure the endpoint reports is an `LLMFailureError`, so callers need no SDK import."""
    api, _ = failing_api(mocker, status_error(status, "model is blocked", sdk_error))

    with pytest.raises(LLMRequestError) as raised:
        api._prompt("some prompt")

    assert isinstance(raised.value, LLMFailureError)


def test_wrapped_endpoint_error_keeps_the_original_error_as_its_cause(mocker: MockerFixture):
    """The SDK error stays reachable, so status codes and the service classifier still work.

    `tests/service_availability.py` walks the cause chain to decide whether a failure is the
    service's fault; flattening it into a plain message would turn outages into hard failures.
    """
    original = status_error(503, "service unavailable")
    api, _ = failing_api(mocker, original)

    with pytest.raises(LLMRequestError) as raised:
        api._prompt("some prompt")

    assert raised.value.__cause__ is original
    assert is_server_problem(raised.value)


def test_wrapped_endpoint_error_message_names_the_model_status_and_reason(mocker: MockerFixture):
    """The status the endpoint returned and its own explanation both reach the caller."""
    api, _ = failing_api(mocker, status_error(503, "model is blocked"))

    with pytest.raises(LLMRequestError) as raised:
        api._prompt("some prompt")

    message = str(raised.value)
    assert api.model in message
    assert "status 503" in message
    assert "model is blocked" in message


def test_connection_errors_are_wrapped_without_a_status(mocker: MockerFixture):
    """A failure with no HTTP status is still described, naming the transport error instead."""
    connection_failure = openai.APIConnectionError(request=httpx.Request("POST", "https://llm.example.invalid/v1"))
    api, _ = failing_api(mocker, connection_failure)

    with pytest.raises(LLMRequestError) as raised:
        api._prompt("some prompt")

    assert "APIConnectionError" in str(raised.value)


def test_endpoint_errors_are_retried_before_failing(mocker: MockerFixture):
    """A request failure is retried like any other, so a momentary 503 need not end the task.

    Documents the cost of one error hierarchy: combined with the SDK's own `max_retries`, a
    hopeless call (bad credentials, blocked model) still costs `prompt_attempts * max_retries`
    requests before it is reported.
    """
    api, create = failing_api(mocker, status_error(503, "service unavailable"))

    with pytest.raises(LLMRequestError):
        api._prompt("some prompt")

    assert create.call_count == TextGenerationAPI.prompt_attempts


def test_exception_messages_never_carry_the_api_key(mocker: MockerFixture):
    """Our own additions to a failure message never include the credential.

    The SDK's message is forwarded intact, so a gateway that quoted the key back would still put
    it in the text; what this pins is that we never interpolate it ourselves.
    """
    empty_api, _ = mocked_api(mocker, None)
    failing, _ = failing_api(mocker, status_error(401, "invalid api key"))

    with pytest.raises(LLMFailureError) as empty:
        empty_api._prompt("some prompt")
    with pytest.raises(LLMRequestError) as failed:
        failing._prompt("some prompt")

    assert MOCK_API_KEY not in str(empty.value)
    assert MOCK_API_KEY not in str(failed.value)
