import os
from pathlib import Path
import pandas as pd
from aoptk.chemical import Chemical
from aoptk.effect import Effect
from aoptk.relationships.relationship import Relationship
from aoptk.relationships.relationship_type import Causation
from aoptk.relationships.relationship_type import Inhibition
from aoptk.text_generation_api import LLMFailureError
from aoptk.text_generation_api import TextGenerationAPI

litellm_api_key = os.getenv("LITELLM_API_KEY")


def write_relationships(publication_id: str, relationships: list[Relationship]) -> None:
    """Writes the relationships to a TSV file."""
    Path("relationships").mkdir(exist_ok=True)
    with Path(f"relationships/{publication_id}.tsv").open("w") as f_out:
        f_out.write("id\tchemical\teffect\trelationship\n")
        f_out.writelines(
            f"{publication_id}\t{relationship.chemical}\t{relationship.effect}\t{relationship.relationship_type}\n"
            for relationship in relationships
        )


def write_chemicals(publication_id: str, chemicals: list[Chemical]) -> None:
    """Writes the chemicals to a TSV file."""
    Path("chemicals").mkdir(exist_ok=True)
    df = pd.DataFrame([chem.to_dict() for chem in chemicals])
    df.to_csv(f"chemicals/{publication_id}.tsv", sep="\t", index=False)


publications = list(Path("publications").iterdir())[:3]
effects = [Effect("liver fibrosis"), Effect("liver cell death")]
relationship_types = [Causation(), Inhibition()]

# TextGenerationAPI already retries transient failures of a single request, so a publication
# that still fails here is recorded and skipped rather than re-queued.
api = TextGenerationAPI(model="gpt-oss-120b", api_key=litellm_api_key)
failed = []

for publication in publications:
    with Path.open(publication) as f_in:
        text = f_in.read()

    publication_id = Path(publication).stem

    try:
        if not Path(f"chemicals/{publication_id}.tsv").exists():
            write_chemicals(publication_id, api.find_chemicals(text))

        chemicals = pd.read_csv(f"chemicals/{publication_id}.tsv", sep="\t")["name"].tolist()
        relationships = TextGenerationAPI(api_key=litellm_api_key).find_relationships_in_text(
            text=text,
            chemicals=[Chemical(name=name) for name in chemicals],
            effects=effects,
            relationship_types=relationship_types,
        )
        write_relationships(publication_id, relationships)
    except LLMFailureError as error:
        print(f"{publication_id}: {error}")
        failed.append(publication)
