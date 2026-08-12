# ArtiFact: A Large-Scale Multi-Modal Cultural Heritage Dataset

**ArtiFact** is a large-scale multi-modal dataset of **651,045** museum records pairing artwork images with structured metadata from three major institutions:

- [The Metropolitan Museum of Art (MET)](https://www.metmuseum.org/)
- [The Art Institute of Chicago (AIC)](https://www.artic.edu/)
- [The Rijksmuseum](https://www.rijksmuseum.nl/)

The dataset supports downstream tasks such as semantic query processing, multi-modal retrieval, and multi-modal error detection. It includes a curated taxonomy of realistic errors with **130K** annotated records for benchmarking data-cleaning methods.

## Quick links

| Resource | Link |
|----------|------|
| Paper | [2606.09648](https://arxiv.org/abs/2606.09648) ([PDF](https://arxiv.org/pdf/2606.09648)) |
| HuggingFace | [deem-data/ArtiFact](https://huggingface.co/datasets/deem-data/ArtiFact) |
<!-- | Google Drive | [Download folder](https://drive.google.com/drive/folders/13cusJy1K5glQpHxqR6cgRn3oxy3CSAaE?usp=share_link) | -->
| Project page | [docs/index.html](docs/index.html) |
| GitHub | [OlgaOvcharenko/ArtiFact](https://github.com/OlgaOvcharenko/ArtiFact) |

## Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("deem-data/ArtiFact")
print(ds)
print(ds["train"][0])
```

## Repository layout

```
ArtiFact/
├── extraction/           # Harvest records from MET, AIC, and Rijksmuseum APIs
├── normalization/        # Rule-based normalization (dates, dimensions, etc.)
├── parsing/              # Base parsing of artists, dates, medium, dimensions
├── LLM_parsing/          # LLM-assisted semantic parsing
├── semantic_unification/ # Vocabulary mapping and schema consolidation
├── error_injection/      # World-knowledge base and error benchmark generation
├── hydra_experiments/    # Downstream experiment scripts (CLIP, AutoGluon, etc.)
├── notebooks/            # Sample exploration notebooks per museum
├── data/                 # Small MET department samples (1,000 rows each)
└── docs/                 # Project website
```

## Reproduce the pipeline

```bash
./1_harvest.sh          # Extract records from museum APIs → csv_pipeline/
./2_normalize.sh        # Normalize raw fields → normalized_pipeline/
./3_parse_base.sh       # Rule-based parsing → parsed_pipeline/
./4_parse_llm.sh        # LLM-assisted parsing (dates, dimensions, artists)
./5_mapping.sh          # Schema consolidation and vocabulary mapping
./6_unify.sh            # Apply semantic unification → 6_unified/
./7_world_knowledge.sh  # Build place/entity knowledge base
./7a_generate_embeddings.sh  # Optional: generate CLIP embeddings for image swap
./8_inject_errors.sh    # Generate error-injected train/test benchmark
```
## Citation

```bibtex
@inproceedings{duarte2026artifact,
  title         = {{ArtiFact}: A Large-Scale Multi-Modal Cultural Heritage Dataset},
  author        = {Duarte, Luciano and Ovcharenko, Olga and Schelter, Sebastian},
  booktitle     = {NOVAS Workshop (Novel Optimizations for Visionary AI Systems) at VLDB 2026},
  year          = {2026},
  url           = {https://arxiv.org/pdf/2606.09648}
}
```

## Authors

- Luciano Duarte — BIFOLD & TU Berlin ([duarte.castineira@tu-berlin.de](mailto:duarte.castineira@tu-berlin.de))
- Olga Ovcharenko — BIFOLD & TU Berlin ([ovcharenko@tu-berlin.de](mailto:ovcharenko@tu-berlin.de))
- Sebastian Schelter — BIFOLD & TU Berlin ([schelter@tu-berlin.de](mailto:schelter@tu-berlin.de))

## License

This repository is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

For licensing inquiries, contact [ovcharenko@tu-berlin.de](mailto:ovcharenko@tu-berlin.de).
