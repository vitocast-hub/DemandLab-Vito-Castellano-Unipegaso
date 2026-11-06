"""Rigenera, in modo ripetibile, il file CSV utilizzato dalla dashboard."""
from pathlib import Path
from src.dataset import generate_dataset

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "domanda_decathlon_simulata.csv"
    generate_dataset(seed=20261105).to_csv(path, index=False)
    print(f"Generati {path.name}: 144 record (dati SIMULATI)")
