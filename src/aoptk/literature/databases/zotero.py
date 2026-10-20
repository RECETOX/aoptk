"""Read literature from a Zotero library (zotero.org or the local desktop app).

Requires the optional dependency ``pyzotero`` (``pip install aoptk[zotero]``).

Example for the local desktop app (Zotero must be running with the local API enabled)::

    Zotero(storage, figure_storage, Query("cadmium"), local=True)
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
import pymupdf
from aoptk.literature.abstract import Abstract
from aoptk.literature.get_abstract import GetAbstract
from aoptk.literature.get_id import GetID
from aoptk.literature.get_metadata import GetMetadata
from aoptk.literature.get_pdf import GetPDF
from aoptk.literature.get_publication import GetPublication
from aoptk.literature.id import DOI
from aoptk.literature.id import ID
from aoptk.literature.id import PMCID
from aoptk.literature.id import PMID
from aoptk.literature.metadata import Metadata
from aoptk.literature.pdf import PDF
from aoptk.literature.publication import Publication
from aoptk.literature.query import Query
from aoptk.literature.utils import convert_image_format

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from pyzotero.errors import PyZoteroError

    _ZOTERO_ERRORS: tuple[type[BaseException], ...] = (PyZoteroError, OSError)
except ImportError:  # pragma: no cover - depends on the optional dependency
    _ZOTERO_ERRORS = (OSError,)

_ITEM_KEY_PATTERN = re.compile(r"^[0-9A-Z]{8}$")
_NON_PUBLICATION_TYPES = ("attachment", "note", "annotation")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _build_client(library_id: str | int, library_type: str, api_key: str | None, local: bool) -> Any:  # noqa: ANN401
    """Create a pyzotero client, importing the optional dependency lazily."""
    try:
        from pyzotero import Zotero as PyZotero  # noqa: PLC0415
    except ImportError as error:
        msg = "Zotero support requires the optional dependency pyzotero - install it with `pip install aoptk[zotero]`."
        raise ImportError(msg) from error
    return PyZotero(library_id=library_id, library_type=library_type, api_key=api_key, local=local)


class Zotero(GetID, GetMetadata, GetAbstract, GetPDF, GetPublication):
    """Class to get data from a Zotero library (local desktop app or zotero.org) based on a query.

    Only read operations are used; the library is never modified.
    """

    unified_image_format = "png"

    def __init__(  # noqa: PLR0913
        self,
        storage: Path,
        figure_storage: Path,
        query: Query | None = None,
        *,
        library_id: str | int | None = None,
        library_type: Literal["user", "group"] = "user",
        api_key: str | None = None,
        collection: str | None = None,
        local: bool = False,
        full_text_search: bool = False,
    ) -> None:
        """Initialize the Zotero database.

        Args:
            storage (Path): Directory for downloaded PDFs and texts.
            figure_storage (Path): Directory for figures.
            query (Query | None): Query used to search the library.
            library_id (str | int | None): Zotero library ID. Required for the remote API, defaults to 0 locally.
            library_type (Literal["user", "group"]): Type of the library. Groups only work with the remote API.
            api_key (str | None): Zotero API key (optional for public libraries and the local API).
            collection (str | None): Key of a collection to restrict the search to.
            local (bool): Use the local Zotero desktop API instead of api.zotero.org.
            full_text_search (bool): Also search inside the full text of attachments.
        """
        if not query:
            query = Query(search_term="")
        if local and library_type == "group":
            msg = "The local Zotero API only serves the local user library, not group libraries."
            raise ValueError(msg)
        if library_id is None:
            if not local:
                msg = "library_id is required when using the remote Zotero API."
                raise ValueError(msg)
            library_id = 0
        self.full_text_search = full_text_search
        self.search_term, self._search_kwargs = self.build_search_term(query)
        self.storage = Path(storage)
        self.figure_storage = Path(figure_storage)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.figure_storage.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self._client = _build_client(library_id, library_type, api_key, local)
        self._items: list[dict] | None = None

    def build_search_term(self, query: Query) -> tuple[str, dict]:
        """Convert a Query to the Zotero search term and search parameters."""
        for name in ("full_text_subset", "only_preprint", "exclude_preprint", "licensing", "date"):
            if getattr(query, name):
                msg = f"The '{name}' filter is not available in Zotero."
                raise NotImplementedError(msg)
        kwargs: dict = {}
        if query.search_term:
            kwargs["q"] = query.search_term
            kwargs["qmode"] = "everything" if self.full_text_search else "titleCreatorYear"
        return query.search_term, kwargs

    def get_ids(self) -> list[ID]:
        """Get IDs (DOI, PMID, PMCID or Zotero item key) of the publications matching the query."""
        return [_get_publication_id(item) for item in self._get_scoped_items()]

    def get_publications_metadata(self, ids: list[ID]) -> list[Metadata]:
        """Retrieve publication metadata."""
        metadata = []
        for item in self._resolve(ids):
            data = item["data"]
            extra = data.get("extra", "")
            doi = data.get("DOI")
            pmid = _extract_identifier(extra, "PMID")
            pmcid = _extract_identifier(extra, "PMCID")
            year = _get_year(data.get("date", ""))
            metadata.append(
                Metadata(
                    id=_get_publication_id(item),
                    doi=DOI(doi) if doi else None,
                    pmid=PMID(pmid) if pmid else None,
                    pmcid=PMCID(pmcid) if pmcid else None,
                    year=int(year) if year else None,
                    title=data.get("title") or None,
                    authors=_get_authors(data.get("creators", [])) or None,
                ),
            )
        return metadata

    def get_abstracts(self, ids: list[ID]) -> list[Abstract]:
        """Retrieve abstracts and write them to storage."""
        abstracts = []
        for item in self._resolve(ids):
            if text := item["data"].get("abstractNote"):
                abstract = Abstract(_get_publication_id(item), text)
                (self.storage / _file_name(abstract.id, "txt")).write_text(text, encoding="utf-8")
                abstracts.append(abstract)
        return abstracts

    def get_pdfs(self, ids: list[ID]) -> list[PDF]:
        """Retrieve the PDF attachments of the publications and store them as `{id}.pdf`."""
        pdfs = []
        for item in self._resolve(ids):
            publication_id = _get_publication_id(item)
            try:
                attachment = self._find_attachment(item, _is_pdf)
                if attachment and (content := self._read_attachment(attachment)) is not None:
                    path = self.storage / _file_name(publication_id, "pdf")
                    path.write_bytes(content)
                    pdfs.append(PDF(path))
            except _ZOTERO_ERRORS:
                continue
        return pdfs

    def get_publications(self, ids: list[ID], download_figures_enabled: bool = True) -> list[Publication]:  # type: ignore[override]
        """Retrieve publications including full text.

        The full text comes from Zotero's full-text index of the PDF attachment. If it is not available
        (for example on the local API), the text is extracted from the PDF attachment itself.
        Tables and figure descriptions are not available from Zotero and stay empty.

        Args:
            ids (list[ID]): A list of publication IDs to retrieve.
            download_figures_enabled (bool): Whether to copy image attachments to the figure storage.
        """
        publications = []
        for item in self._resolve(ids):
            try:
                publication = self._get_publication(item, download_figures_enabled=download_figures_enabled)
            except _ZOTERO_ERRORS:
                continue
            if publication:
                (self.storage / _file_name(publication.id, "txt")).write_text(publication.full_text, encoding="utf-8")
                publications.append(publication)
        return publications

    def _get_publication(self, item: dict, *, download_figures_enabled: bool) -> Publication | None:
        publication_id = _get_publication_id(item)
        abstract = Abstract(publication_id, item["data"].get("abstractNote", ""))
        pdf_attachment = self._find_attachment(item, _is_pdf)
        full_text = self._get_full_text(pdf_attachment) if pdf_attachment else ""
        if not full_text:
            return None
        figures = self._get_figures(item, publication_id) if download_figures_enabled else []
        return Publication(publication_id, abstract, full_text, [], figures, [])

    def _get_full_text(self, attachment: dict) -> str:
        try:
            content = self._client.fulltext_item(attachment["key"]).get("content", "")
            if content:
                return content
        except _ZOTERO_ERRORS:
            pass
        pdf_bytes = self._read_attachment(attachment)
        if not pdf_bytes:
            return ""
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document)

    def _get_figures(self, item: dict, publication_id: ID) -> list[Path]:
        figures = []
        for index, attachment in enumerate(self._children(item)):
            filename = attachment["data"].get("filename", "")
            if attachment["data"].get("itemType") != "attachment" or not filename.lower().endswith(_IMAGE_EXTENSIONS):
                continue
            content = self._read_attachment(attachment)
            if content is None:
                continue
            path = self.figure_storage / _file_name(publication_id, Path(filename).suffix.lower(), f"_{index}")
            path.write_bytes(content)
            figures.append(path)
        return convert_image_format(figures, self.unified_image_format)

    def _children(self, item: dict) -> list[dict]:
        return self._client.children(item["key"])

    def _find_attachment(self, item: dict, predicate: Callable[[dict], bool]) -> dict | None:
        for child in self._children(item):
            data = child["data"]
            if data.get("itemType") == "attachment" and predicate(data):
                return child
        return None

    def _read_attachment(self, attachment: dict) -> bytes | None:
        data = attachment["data"]
        if data.get("linkMode") == "linked_file" and data.get("path"):
            linked = Path(data["path"])
            return linked.read_bytes() if linked.is_file() else None
        return self._client.file(attachment["key"])

    def _get_scoped_items(self) -> list[dict]:
        """Fetch (once) the top-level items of the library or collection matching the query."""
        if self._items is None:
            if self.collection:
                items = self._client.everything(
                    self._client.collection_items_top(self.collection, **self._search_kwargs),
                )
            else:
                items = self._client.everything(self._client.top(**self._search_kwargs))
            self._items = [
                item
                for item in items
                if not item["data"].get("parentItem") and item["data"].get("itemType") not in _NON_PUBLICATION_TYPES
            ]
        return self._items

    def _resolve(self, ids: list[ID]) -> list[dict]:
        """Map requested IDs to Zotero items, skipping IDs that cannot be found."""
        resolved = []
        for requested in ids:
            found = next((item for item in self._get_scoped_items() if _matches(item, requested)), None)
            if found is None and _ITEM_KEY_PATTERN.match(str(requested)):
                try:
                    found = self._client.item(str(requested))
                except _ZOTERO_ERRORS:
                    found = None
            if found is not None and found["data"].get("itemType") not in _NON_PUBLICATION_TYPES:
                resolved.append(found)
        return resolved


def _file_name(publication_id: ID, extension: str, suffix: str = "") -> str:
    """Build a file name from an ID; DOIs contain path separators that must be replaced."""
    extension = extension.lstrip(".")
    return f"{re.sub(r'[^A-Za-z0-9._-]', '_', str(publication_id))}{suffix}.{extension}"


def _is_pdf(data: dict) -> bool:
    return data.get("contentType") == "application/pdf" or data.get("filename", "").lower().endswith(".pdf")


def _matches(item: dict, requested: ID) -> bool:
    data = item["data"]
    extra = data.get("extra", "")
    candidates = (data.get("DOI"), _extract_identifier(extra, "PMID"), _extract_identifier(extra, "PMCID"), item["key"])
    return str(requested) in [c for c in candidates if c]


def _extract_identifier(extra: str, label: str) -> str | None:
    """Extract an identifier such as 'PMID: 123' from Zotero's free-text 'extra' field."""
    match = re.search(rf"^\s*{label}\s*:\s*(\S+)", extra or "", flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else None


def _get_publication_id(item: dict) -> ID:
    """Return the DOI, PMID, PMCID or Zotero key (in this order) of an item."""
    data = item["data"]
    extra = data.get("extra", "")
    publication_id = (
        data.get("DOI")
        or _extract_identifier(extra, "PMID")
        or _extract_identifier(extra, "PMCID")
        or item.get("key")
        or data.get("key")
    )
    if not publication_id:
        msg = "Zotero item is missing a publication id"
        raise ValueError(msg)
    return ID(publication_id)


def _get_year(date: str) -> str | None:
    match = re.search(r"\d{4}", date or "")
    return match.group(0) if match else None


def _get_authors(creators: list[dict]) -> list[str]:
    authors = []
    for creator in creators:
        if creator.get("creatorType", "author") != "author":
            continue
        if last := creator.get("lastName"):
            initials = "".join(f"{part[0]}." for part in creator.get("firstName", "").replace("-", " ").split())
            authors.append(f"{last} {initials}".strip())
        elif name := creator.get("name"):
            authors.append(name)
    return authors
