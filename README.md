[![github license badge](https://img.shields.io/github/license/RECETOX/aoptk)](https://github.com/RECETOX/aoptk) [![workflow pypi badge](https://img.shields.io/pypi/v/aoptk.svg?colorB=blue)](https://pypi.python.org/project/aoptk/) [![Bioconda](https://anaconda.org/bioconda/aoptk/badges/version.svg)](https://anaconda.org/bioconda/aoptk) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20036704.svg)](https://doi.org/10.5281/zenodo.20036704) [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=RECETOX_aoptk&metric=alert_status)](https://sonarcloud.io/dashboard?id=RECETOX_aoptk) [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=RECETOX_aoptk&metric=coverage)](https://sonarcloud.io/dashboard?id=RECETOX_aoptk) [![Documentation Status](https://readthedocs.org/projects/aoptk/badge/?version=latest)](https://aoptk.readthedocs.io/en/latest/?badge=latest) [![build](https://github.com/RECETOX/aoptk/actions/workflows/build.yml/badge.svg)](https://github.com/RECETOX/aoptk/actions/workflows/build.yml) [![cffconvert](https://github.com/RECETOX/aoptk/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/RECETOX/aoptk/actions/workflows/cffconvert.yml) [![sonarcloud](https://github.com/RECETOX/aoptk/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/RECETOX/aoptk/actions/workflows/sonarcloud.yml)

![AOPTK Banner](docs/aoptk_banner_slim.png)
## AOP-toolkit (aoptk)  
AOP-toolkit (aoptk) is a Python package for mining and analyzing toxicological and biomedical literature. Originally developed to support the construction of Adverse Outcome Pathways (AOPs), it provides general-purpose tools for retrieving, processing, and analyzing scientific publications.

The toolkit enables users to collect literature from databases such as PubMed Central and Europe PMC, extract information from full-text articles, analyze unstructured text and images using large language models, and normalize chemical names across publications to improve data consistency and interoperability.

## How to use aoptk


## Installation

You can install aoptk in multiple ways depending on your workflow.

### Install from PyPI (with uv)

```console
python -m pip install aoptk
```

### Install from BioConda

```console
conda install -c bioconda aoptk
```

### Install from source (with uv)

```console
git clone git@github.com:RECETOX/aoptk.git
cd aoptk
uv sync --frozen
uv pip install .
```

### Dev Container (developer environment)

This repo includes a lightweight VS Code Dev Container in the `.devcontainer/` folder. Rebuild it via VS Code: "Dev Containers: Rebuild Container".

To configure API access, copy `.env.example` to `.env` and fill in the required values before rebuilding the container.

## Documentation

See our [Read the Docs](https://aoptk.readthedocs.io/en/latest/).

## Contributing

If you want to contribute to the development of aoptk,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## Funding

This software was developed with support from the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 857560 (CETOCOEN Excellence) and from MEYS under grant SALVAGE (CZ.02.01.01/00/22_008/0004644). The content reflects only the author’s view, and the European Commission is not responsible for any use that may be made of the information it contains. Authors also thank the RECETOX Research Infrastructure (No LM2023069) financed by the Ministry of Education, Youth and Sports for supportive background.

