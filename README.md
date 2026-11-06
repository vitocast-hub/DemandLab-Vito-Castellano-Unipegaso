# DemandLab — Previsione della domanda e gestione delle scorte

DemandLab è una dashboard interattiva sviluppata in Python con Streamlit per analizzare l'andamento della domanda, confrontare metodi previsionali e simulare la pianificazione delle scorte.

## Dati utilizzati

L'applicazione considera quattro categorie di articoli sportivi, prendendo come riferimento merceologico l'assortimento Decathlon:

- Scarpe da running: domanda relativamente stabile.
- Maschere da snorkeling: domanda con stagionalità estiva.
- Giacche termiche: domanda con stagionalità invernale.
- Tappetini fitness: domanda con tendenza crescente.

Il dataset comprende **144 osservazioni mensili simulate**, relative a quattro articoli e a 36 mesi (da novembre 2023 a ottobre 2026). I dati sono generati in modo riproducibile e **non rappresentano vendite, giacenze o previsioni ufficiali di Decathlon**. L'applicazione è indipendente dall'azienda.

## Funzionalità

- Visualizzazione della domanda storica e della previsione per i sei mesi successivi.
- Confronto tra cinque metodi: media mobile semplice, attenuazione esponenziale semplice, regressione lineare, metodo *naive* stagionale e regressione con componenti stagionali.
- Valutazione dell'accuratezza su 12 mesi mediante MAE, RMSE e sMAPE, con selezione del metodo migliore in base al MAE (e RMSE in caso di parità).
- Calcolo delle scorte cicliche, di sicurezza, anticipate e in transito (*pipeline*).
- Simulazione di scenari modificando trend, stagionalità, variabilità della domanda, tempi di approvvigionamento, intervallo di riordino, livello di servizio e capacità di rifornimento.
- Grafici interattivi, confronto tra scenario di riferimento e scenario modificato ed esportazione dei risultati in formato CSV.

## Requisiti

- Python 3.11, 3.12 o 3.13.
- Librerie elencate in `requirements.txt`.
- Connessione a Internet per installare le librerie al primo avvio.

## Avvio in locale

Aprire un terminale nella cartella del progetto, dove si trova `app.py`.

**Windows:**

```cmd
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

**macOS / Linux:**

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

La dashboard si apre nel browser all'indirizzo indicato dal terminale, normalmente `http://localhost:8501`. Per arrestarla, premere `Ctrl+C` nel terminale.

## Rigenerazione dei dati

Per ricostruire il dataset simulato con lo stesso seme casuale:

```bash
python generate_data.py
```

Il file generato si trova in `data/domanda_decathlon_simulata.csv`.

## Test automatici

I test controllano la riproducibilità dei dati e la coerenza dei calcoli previsionali e delle scorte. Per eseguirli:

```bash
python -m pip install pytest
python -m pytest -q
```

Su Windows, se `python` non è riconosciuto, utilizzare `py` al suo posto.

## Struttura del progetto

```text
├── .streamlit/
│   └── config.toml
├── data/
│   └── domanda_decathlon_simulata.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── forecast.py
│   ├── inventory.py
│   └── scenarios.py
├── tests/
│   └── test_models.py
├── .gitignore
├── app.py
├── generate_data.py
├── README.md
└── requirements.txt
```

## Pubblicazione facoltativa su Streamlit Community Cloud

Dopo avere caricato **i file e le cartelle del progetto**, mantenendone la struttura, in una repository GitHub, è possibile pubblicare la dashboard su [Streamlit Community Cloud](https://share.streamlit.io). Durante la configurazione occorre selezionare la repository, il ramo corretto e `app.py` come file principale. Le librerie vengono installate a partire da `requirements.txt`.

## Interpretazione dei risultati

Le simulazioni illustrano gli effetti delle ipotesi scelte sui fabbisogni di scorta; non costituiscono indicazioni operative basate su dati aziendali effettivi. La scorta *pipeline* rappresenta prodotti in transito, non merce già disponibile a magazzino. Le componenti sono confrontate a fini analitici e la loro somma non equivale automaticamente alla giacenza fisica o alla quantità da ordinare.
