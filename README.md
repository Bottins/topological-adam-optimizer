# Topological-Adam Optimizer

Implementazione sperimentale di `TAdam`, una variante di Adam con scaling del learning rate basato su metriche topologiche del grafo.

## Contenuto

- `tadam.py`: ottimizzatore TAdam
- `test_tadam.py`: test suite/benchmark su grafi sintetici
- `Adam_v/`: varianti Adam di riferimento

## Setup

```bash
pip install -r requirements.txt
```

Dipendenze opzionali:

- `torch-geometric` (alcuni test/modelli)

## Esecuzione test

```bash
python test_tadam.py
```

Output principale:

- `tadam_comparison.png`

## Nota

Il file `test_tadam.py` e uno script dimostrativo completo (esegue direttamente i test).
## Research Profile

- Research keywords: graph optimization, adaptive learning rates, topology-aware training, GNN optimization.
- Positioning: methodological research project on optimizer design.
- Open-source status: this repository is open source and intended for reproducible research and education.

