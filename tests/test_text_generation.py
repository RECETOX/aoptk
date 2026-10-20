from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import pytest
from flaky import flaky
from aoptk.chemical import Chemical
from aoptk.effect import Effect
from aoptk.find_chemical import FindChemical
from aoptk.literature.convert_image import ConvertImage
from aoptk.literature.convert_pdf_scan import ConvertPDFScan
from aoptk.literature.find_relevant_publication import FindRelevantPublication
from aoptk.normalization.normalize_chemical import NormalizeChemical
from aoptk.relationships.find_relationship import FindRelationship
from aoptk.relationships.relationship import Relationship
from aoptk.relationships.relationship_type import Causation
from aoptk.relationships.relationship_type import Inhibition
from aoptk.relationships.relationship_type import RelationshipType
from aoptk.text_generation_api import DEFAULT_VISION_MODEL
from aoptk.text_generation_api import TextGenerationAPI
from tests.service_availability import require_llm_service


@pytest.fixture(autouse=True)
def require_llm() -> None:
    """Require the text generation endpoint before running any test of this module."""
    require_llm_service()


def sort_key(r: Relationship) -> tuple[str, str, str]:
    """Sort key for Relationship objects.

    Args:
        r (Relationship): Relationship object.
    """
    return (r.relationship_type, r.chemical.name, r.effect.name)


@pytest.mark.openai
def test_can_create():
    """Can create ScispacyFindChemical instance."""
    actual = TextGenerationAPI()
    assert actual is not None


@pytest.mark.openai
def test_implements_interface_find_chemical():
    """ScispacyFindChemical implements FindChemical interface."""
    assert isinstance(TextGenerationAPI(), FindChemical)
    assert isinstance(TextGenerationAPI(), FindRelationship)
    assert isinstance(TextGenerationAPI(), NormalizeChemical)
    assert isinstance(TextGenerationAPI(), ConvertPDFScan)
    assert isinstance(TextGenerationAPI(), ConvertImage)
    assert isinstance(TextGenerationAPI(), FindRelevantPublication)


@pytest.mark.openai
def test_find_chemical_not_empty():
    """Test that find_chemical method returns a non-empty result."""
    actual = TextGenerationAPI().find_chemicals("")
    assert actual is not None


@pytest.mark.openai
def test_find_relationships_not_empty():
    """Test that find_relationships method returns a non-empty result."""
    actual = TextGenerationAPI().find_relationships_in_text(text="", relationship_types=[], chemicals=[], effects=[])
    assert actual is not None


@pytest.mark.openai
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Thioacetamide and ethanol were used to induce liver injury.",
            ["thioacetamide", "ethanol"],
        ),
        (
            "Thioacetamide was used to induce liver injury.",
            ["thioacetamide"],
        ),
        (
            "No chemical was studied here.",
            [],
        ),
    ],
)
def test_find_chemical(text: str, expected: list[str]):
    """Test that find_chemical method finds chemicals in text."""
    actual = [chem.name for chem in TextGenerationAPI().find_chemicals(text)]
    assert actual == expected


@flaky(max_runs=3, min_passes=1)
@pytest.mark.openai
@pytest.mark.parametrize(
    ("text", "relationship_types", "chemicals", "effects", "expected_relationships"),
    [
        (
            "Cancer is caused by thioacetamide, not by acetaminophen.",
            [Causation()],
            [Chemical(name="acetaminophen"), Chemical(name="thioacetamide")],
            [Effect(name="cancer")],
            [
                Relationship(
                    relationship_type=Causation().negative,
                    chemical=Chemical(name="acetaminophen"),
                    effect=Effect(name="cancer"),
                    context="Cancer is caused by thioacetamide, not by acetaminophen.",
                ),
                Relationship(
                    relationship_type=Causation().positive,
                    chemical=Chemical(name="thioacetamide"),
                    effect=Effect(name="cancer"),
                    context="Cancer is caused by thioacetamide, not by acetaminophen.",
                ),
            ],
        ),
        (
            "Just some random text with no effect and no chemical in here.",
            [Causation()],
            [],
            [],
            [],
        ),
        (
            "Thioacetamide was studied. Acetaminophen caused liver fibrosis.",
            [Causation(), Inhibition()],
            [Chemical(name="thioacetamide"), Chemical(name="acetaminophen")],
            [Effect(name="liver fibrosis"), Effect(name="cancer")],
            [
                Relationship(
                    relationship_type=Causation().positive,
                    chemical=Chemical(name="acetaminophen"),
                    effect=Effect(name="liver fibrosis"),
                    context="Thioacetamide was studied. Acetaminophen caused liver fibrosis.",
                ),
            ],
        ),
    ],
)
def test_find_relationships(
    text: str,
    relationship_types: list[RelationshipType],
    chemicals: list[Chemical],
    effects: list[Effect],
    expected_relationships: list[Relationship],
):
    """Test find_relationships method with multiple chemicals and effects."""
    actual = TextGenerationAPI().find_relationships_in_text(
        text=text,
        relationship_types=relationship_types,
        chemicals=chemicals,
        effects=effects,
    )

    assert sorted(actual, key=sort_key) == sorted(expected_relationships, key=sort_key)


@pytest.fixture
def phthalate_table_data():
    """Provide a sample phthalate table payload for tests."""
    return {
        "Carbon Chain Length.": [
            "Short",
            "Long",
        ],
        "GJIC.EC 50 b ( µ M) 80": [
            "70",
            "> 200",
        ],
        "Phthalate.": [
            "Dipropyl phthalate",
            "Diisodecyl phthalate",
        ],
    }


@pytest.mark.openai
@flaky
def test_relationship_table(phthalate_table_data: dict):
    """Test find_relationships_in_table method with a table."""
    actual = TextGenerationAPI().find_relationships_in_table(
        table_df=pd.DataFrame(phthalate_table_data),
        relationship_types=[Inhibition()],
        effects=[Effect(name="gap junction intercellular communication")],
    )
    assert any(
        r.chemical.name == "dipropyl phthalate"
        and r.effect.name == "gap junction intercellular communication"
        and r.relationship_type == Inhibition().positive
        and r.context == "table"
        for r in actual
    )


@pytest.mark.openai
@pytest.mark.parametrize(
    ("chemical", "list_of_chemicals", "expected_heading"),
    [
        (
            Chemical(name="paracetamol"),
            [Chemical(name="acetaminophen"), Chemical(name="thioacetamide")],
            "acetaminophen",
        ),
    ],
)
def test_normalize_chemical(chemical: Chemical, list_of_chemicals: list[Chemical], expected_heading: str):
    """Test that find_chemical method finds chemicals in text."""
    actual = TextGenerationAPI().normalize_chemical(chemical, list_of_chemicals)
    assert actual.heading == expected_heading


@pytest.mark.openai
def test_extract_text_from_pdf_image():
    """Test that extract_text_from_pdf_image method extracts text from a PDF image."""
    base64_str = (Path("tests/test_data/scan_base64_image_PMC12416454.txt").read_text()).strip()
    actual = TextGenerationAPI(model=DEFAULT_VISION_MODEL).convert_pdf_scan(
        base64_str,
        mime_type="image/jpeg",
    )
    assert ("More than three decades after the first discovery, most") in actual


@pytest.mark.openai
@pytest.mark.parametrize(
    ("text", "images", "expected_chemicals"),
    [
        (
            "Gap junction intracellular communication was studied in this study.",
            ["tests/test_data/test_figures/gjic.jpeg"],
            ["dibutyl phthalate"],
        ),
    ],
)
def test_find_relationships_in_text_and_images(text: str, images: list[str], expected_chemicals: list[str]):
    """Test that find_relationships_in_text_and_images method finds relationships in text and images."""
    actual = TextGenerationAPI(model=DEFAULT_VISION_MODEL).find_relationships_in_text_and_images(
        text=text,
        image_paths=images,
        relationship_types=[Inhibition()],
        effects=[Effect(name="gap junction intercellular communication")],
    )

    if not expected_chemicals:
        assert len(actual) == 0
    else:
        for expected_chemical in expected_chemicals:
            assert any(
                r.chemical.name == expected_chemical
                and r.effect.name == "gap junction intercellular communication"
                and r.relationship_type == Inhibition().positive
                for r in actual
            )


@pytest.mark.openai
def test_convert_image_to_text():
    """Test that convert_image_to_text method converts an image to text."""
    actual = TextGenerationAPI(model=DEFAULT_VISION_MODEL).convert_image(
        "tests/test_data/test_figures/gjic.jpeg",
        text="These images are about gap junction intercellular communication.",
    )
    assert "gjic" in actual.lower()


@pytest.mark.openai
@pytest.mark.parametrize(
    ("question", "text", "expected"),
    [
        (
            "Does this publication contain data about drug carriers in HepG2 cells?",
            "Bile acid-based drug carriers were studied here. "
            "Both HepG2 and HepaRG spheroids were used as an experimental model.",
            True,
        ),
        (
            "Does this publication contain data about drug carriers in HeLa cells?",
            "Bile acid-based drug carriers were studied here. "
            "Both HepG2 and HepaRG spheroids were used as an experimental model.",
            False,
        ),
    ],
)
def test_find_relevant_publications(question: str, text: str, expected: bool):
    """Test that find_relevant_publications method finds relevant publications."""
    actual = TextGenerationAPI().find_relevant_publications(question=question, text=text)
    assert actual == expected


@pytest.mark.openai
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip on CI environment due to resource constraints.")
def test_retry_strategy_works():
    """Test that the retry strategy works."""
    problematic_text = Path("tests/test_data/PMC11780512.txt").read_text()
    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(
            executor.map(
                lambda _: TextGenerationAPI().find_chemicals(text=problematic_text),
                range(1, num_threads + 1),
            ),
        )
    assert len(results) == num_threads


@pytest.mark.parametrize(
    ("text", "invalid_response_patterns", "expected"),
    [
        (
            Path("tests/test_data/invalid_chemical_response.txt").read_text(encoding="utf-8"),
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            """I need to extract chemicals from text...

            thioacetamide
            methotrexate""",
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            """- thioacetamide
            - methotrexate""",
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            "thioacetamide ; methotrexate ; acetaminophen",
            TextGenerationAPI.invalid_chemical_response_patterns,
            False,
        ),
        (
            "thioacetamide",
            TextGenerationAPI.invalid_chemical_response_patterns,
            False,
        ),
        (
            "PCB 123",
            TextGenerationAPI.invalid_chemical_response_patterns,
            False,
        ),
        (
            "PCB-123",
            TextGenerationAPI.invalid_chemical_response_patterns,
            False,
        ),
        (
            "PCB-123 ; thioacetamide ; 3-(4,5-dimethylthiazol-2-yl)-2,5-diphenyltetrazolium bromide",
            TextGenerationAPI.invalid_chemical_response_patterns,
            False,
        ),
        (
            """1. thioacetamide
            2. methotrexate""",
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            """* thioacetamide
            * methotrexate""",
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            """** thioacetamide
            ** methotrexate""",
            TextGenerationAPI.invalid_chemical_response_patterns,
            True,
        ),
        (
            "thioacetamide",
            TextGenerationAPI.invalid_normalization_response_patterns,
            False,
        ),
        (
            "thioacetamide ; methotrexate ; acetaminophen",
            TextGenerationAPI.invalid_normalization_response_patterns,
            True,
        ),
    ],
)
def test_invalid_response(text: str, invalid_response_patterns: list[str], expected: bool):
    """Test that the find_chemicals method handles invalid chemical responses gracefully."""
    actual = TextGenerationAPI().is_invalid_response(text, invalid_response_patterns)
    assert actual == expected


@pytest.mark.parametrize(
    ("text", "categories", "expected"),
    [
        (
            "This text is about liver.",
            ["Liver", "Heart", "Kidney"],
            "liver",
        ),
        (
            "This text is not about any organ.",
            ["Liver", "Heart", "Kidney"],
            None,
        ),
    ],
)
def test_categorize_text(text: str, categories: list[str], expected: str | None):
    """Test that categorize_text method categorizes text correctly."""
    actual = TextGenerationAPI().categorize_text(text=text, categories=categories)
    assert actual == expected
