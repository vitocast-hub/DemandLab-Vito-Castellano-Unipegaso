"""DemandLab — Dashboard di previsione della domanda e gestione delle scorte.

Avvio locale: streamlit run app.py
Dati esclusivamente simulati, non riconducibili a vendite o disponibilità Decathlon.
"""
from __future__ import annotations

from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataset import PRODUCTS, load_dataset
from src.forecast import METHODS, evaluate_methods, forecast_series
from src.inventory import InventoryResult, calculate_inventory
from src.scenarios import transform_forecast

DATA_PATH = Path(__file__).resolve().parent / "data" / "domanda_decathlon_simulata.csv"
INDIGO = "#5F6CE0"
TEAL = "#19B7AA"
AMBER = "#EEAB50"
PALE = "#AAB5D0"
BG = "#12192A"
STOCK_NAMES = {
    "cyclic": "Scorta ciclica",
    "safety": "Scorta di sicurezza",
    "anticipation": "Scorta anticipata",
    "pipeline": "Scorta pipeline (in transito)",
}
STOCK_COLORS = {"cyclic": "#5F6CE0", "safety": "#19B7AA",
                "anticipation": "#EEAB50", "pipeline": "#93A3BD"}

st.set_page_config(page_title="DemandLab", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp {background: #0C1422; color: #EAF1FF;}
    .block-container {padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1400px;}
    [data-testid="stSidebar"] {background: #121D30;}
    [data-testid="stMetric"] {background: #172438; border: 1px solid #2C3A50;
        padding: 16px 18px; border-radius: 14px;}
    [data-testid="stMetricLabel"] {color:#BACBE0;}
    /* Le quattro schede principali non tagliano più i nomi lunghi. */
    .kpi-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:16px; margin:16px 0 8px;}
    .kpi-card {background:#172438; border:1px solid #2C3A50;
        border-radius:14px; padding:17px 18px; min-width:0; min-height:122px;}
    .kpi-label {color:#BACBE0; font-size:.89rem; line-height:1.35; margin-bottom:10px;}
    .kpi-value {color:#EAF1FF; font-size:clamp(1.12rem,1.6vw,1.75rem);
        line-height:1.22; font-weight:600; white-space:normal; overflow-wrap:break-word;}
    .kpi-delta {display:inline-block; margin-top:10px; padding:3px 9px;
        border-radius:999px; background:#173E38; color:#52E6AB;
        font-size:.78rem; font-weight:600;}
    .kpi-delta.neg {background:#462A35; color:#FF8494;}
    .kpi-delta.neutral {background:#28344A; color:#C0CBE0;}
    @media(max-width:1120px) {.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
    @media(max-width:640px) {.kpi-grid{grid-template-columns:1fr;}}
    .banner {padding: 12px 17px; margin: 12px 0 18px; border-radius: 12px;
        background: #172A39; color:#D1EBEB; border:1px solid #2A5360;}
    h1,h2,h3 {letter-spacing:-.025em;}
    [data-testid="stDataFrame"] {border-radius:12px;}
    div[data-testid="stExpander"] {border:1px solid #273A54;border-radius:12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data

def read_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data

def analyze_product(name: str, data_frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    product = data_frame.loc[data_frame["articolo"] == name].sort_values("mese")
    history = pd.Series(product["domanda_unita"].to_numpy(dtype=float),
                        index=pd.DatetimeIndex(product["mese"]), name="Domanda osservata")
    scores, validation = evaluate_methods(history, 12)
    return history, scores, validation


def inferred_cv(scores: pd.DataFrame, validation: pd.DataFrame) -> float:
    """Prudential proxy for monthly residual CV, never claimed as estimated demand law."""
    mean_actual = float(validation["Domanda effettiva"].mean())
    rmse_best = float(scores.iloc[0]["RMSE"])
    return float(np.clip(rmse_best / max(mean_actual, 1.0), .04, .60))


def inventory_for(forecast: pd.Series, cv: float, lead: int, lead_cv: int,
                  review: int, service: float, capacity_monthly: float) -> InventoryResult:
    return calculate_inventory(forecast, cv, lead,
                               lead * lead_cv / 100., review,
                               service / 100., capacity_monthly)


def plot_style(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#D9E7FA", size=12), margin=dict(l=16, r=16, b=12, t=35),
        height=height, hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(gridcolor="#24344E", zeroline=False),
        yaxis=dict(gridcolor="#24344E", zeroline=False),
    )
    return fig


def demand_chart(history: pd.Series, original: pd.Series,
                 simulated: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index[-24:], y=history.iloc[-24:],
                             mode="lines+markers", name="Storico (non cambia)",
                             line=dict(color=PALE, width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[history.index[-1], *original.index],
                             y=[history.iloc[-1], *original.values],
                             mode="lines+markers", name="Riferimento (non cambia)",
                             line=dict(color=INDIGO, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=[history.index[-1], *simulated.index],
                             y=[history.iloc[-1], *simulated.values],
                             mode="lines+markers", name="Scenario (si aggiorna)",
                             line=dict(color=TEAL, width=3), marker=dict(size=7)))
    fig.add_vline(x=history.index[-1].timestamp() * 1000,
                  line_width=1, line_dash="dash", line_color="#8496B1")
    fig.update_yaxes(title_text="Unità al mese", rangemode="normal")
    return plot_style(fig, 410)


def validation_chart(validation: pd.DataFrame, best: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=validation.index, y=validation["Domanda effettiva"],
                             mode="lines+markers", name="Domanda effettiva",
                             line=dict(color=PALE, width=2)))
    fig.add_trace(go.Scatter(x=validation.index, y=validation[best],
                             mode="lines+markers", name=f"Predetto: {best}",
                             line=dict(color=TEAL, width=3)))
    fig.update_yaxes(title_text="Unità al mese", rangemode="tozero")
    return plot_style(fig, 360)


def inventory_chart(reference: InventoryResult, scenario: InventoryResult) -> go.Figure:
    fig = go.Figure()
    for key, name in STOCK_NAMES.items():
        fig.add_trace(go.Bar(
            name=name, x=["Riferimento", "Scenario"],
            y=[getattr(reference, key), getattr(scenario, key)],
            marker_color=STOCK_COLORS[key],
            hovertemplate="%{x}<br>%{y:,.0f} unità<extra>" + name + "</extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Unità (componenti teoriche)", rangemode="tozero")
    return plot_style(fig, 410)


def stock_rows(base: InventoryResult, scenario: InventoryResult) -> pd.DataFrame:
    data = [{"Componente": display, "Riferimento": round(getattr(base, key), 1),
             "Scenario": round(getattr(scenario, key), 1),
             "Differenza": round(getattr(scenario, key) - getattr(base, key), 1)}
            for key, display in STOCK_NAMES.items()]
    return pd.DataFrame(data)


def csv_content(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def kpi_card(label: str, value: str, delta: str = "", trend: int = 0) -> str:
    """Scheda sintetica con testo completo su più righe e variazione facoltativa."""
    badge = ""
    if delta:
        category = "" if trend > 0 else "neg" if trend < 0 else "neutral"
        badge = f'<div class="kpi-delta {category}">{escape(delta)}</div>'
    return (f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div>{badge}</div>')


# Sidebar — one set of controls drives every scenario graphic and stock component.
st.sidebar.markdown('<div class="eyebrow">CONTROLLO SCENARI</div>',
                    unsafe_allow_html=True)
st.sidebar.title("Parametri di simulazione")
selected = st.sidebar.selectbox("Articolo", list(PRODUCTS), index=1)
st.sidebar.caption(PRODUCTS[selected]["description"])
st.sidebar.divider()
st.sidebar.subheader("Domanda")
trend_pct = st.sidebar.slider("Variazione aggiuntiva del trend (% al mese)",
                              min_value=-5., max_value=5., value=0., step=.5)
seasonality_pct = st.sidebar.slider("Intensità stagionale (% della base)",
                                    min_value=0, max_value=200, value=100, step=10)
volatility_multiplier = st.sidebar.slider("Moltiplicatore variabilità domanda",
                                          min_value=.5, max_value=2., value=1., step=.1)
st.sidebar.divider()
st.sidebar.subheader("Approvvigionamento")
lead_days = st.sidebar.slider("Lead time medio (giorni)", 7, 90, 30)
lead_cv = st.sidebar.slider("Deviazione standard lead time (% della media)",
                            0, 40, 10, 5)
review_days = st.sidebar.select_slider("Intervallo di riordino (giorni)",
                                      options=[7, 14, 21, 30, 45, 60], value=30)
service_pct = st.sidebar.select_slider("Livello di servizio obiettivo (%)",
                                       options=[90.0, 95.0, 97.5, 99.0], value=95.0)
capacity_factor = st.sidebar.slider("Capacità di rifornimento (% domanda base media)",
                                    70, 160, 110, 5) / 100.
st.sidebar.divider()
st.sidebar.caption("Trend e stagionalità modificano la curva turchese. "
                   "La variabilità modifica invece la scorta di sicurezza; "
                   "i parametri di approvvigionamento agiscono sulle scorte, "
                   "non sulla domanda storica.")

# Dataset and core analyses.
data = read_data()
history, scores, validation = analyze_product(selected, data)
best = str(scores.loc[0, "Metodo"])
base_forecast = forecast_series(history, best, 6)
scenario_forecast = transform_forecast(base_forecast, history, trend_pct, seasonality_pct)
cv = inferred_cv(scores, validation)
baseline_capacity = float(base_forecast.mean()) * 1.10
scenario_capacity = float(base_forecast.mean()) * capacity_factor
base_stock = inventory_for(base_forecast, cv, 30, 10, 30, 95.0, baseline_capacity)
scenario_stock = inventory_for(scenario_forecast, cv * volatility_multiplier,
                               lead_days, lead_cv, review_days, service_pct,
                               scenario_capacity)

st.title("DemandLab | Previsione domanda e scorte")
st.markdown('<div class="banner"><strong>Origine dei dati:</strong> 144 osservazioni '
            '(4 articoli × 36 mesi) interamente simulate e riproducibili. '
            'Non sono dati di vendita o di magazzino ufficiali di Decathlon.</div>',
            unsafe_allow_html=True)

forecast_change = 100 * (scenario_forecast.mean() / max(base_forecast.mean(), 1) - 1)
safety_change = scenario_stock.safety - base_stock.safety
kpis = [
    kpi_card("Prodotto analizzato", selected),
    kpi_card("Metodo migliore (MAE)", best),
    kpi_card("Domanda media prevista", f"{scenario_forecast.mean():,.0f} un./mese",
             f"{forecast_change:+.1f}% rispetto alla base", 1 if forecast_change > 0 else -1 if forecast_change < 0 else 0),
    kpi_card("Scorta di sicurezza", f"{scenario_stock.safety:,.0f} unità",
             f"{safety_change:+,.0f} unità rispetto alla base", 1 if safety_change > 0 else -1 if safety_change < 0 else 0),
]
st.markdown('<div class="kpi-grid">' + ''.join(kpis) + '</div>', unsafe_allow_html=True)

st.divider()
tab_overview, tab_models, tab_inventory, tab_method = st.tabs([
    "📊 Panoramica e domanda", "🎯 Metodi previsionali", "📦 Scorte e scenari", "📖 Metodo e dati"
])

with tab_overview:
    st.subheader("Andamento storico e proiezione a sei mesi")
    final_delta = float(scenario_forecast.iloc[-1] - base_forecast.iloc[-1])
    st.markdown(
        f"**Scenario attuale:** trend {trend_pct:+.1f}% al mese · "
        f"stagionalità {seasonality_pct}% · "
        f"**differenza nel sesto mese: {final_delta:+,.0f} unità** rispetto alla previsione di base."
    )
    st.caption("Grigio: domanda storica invariata. Blu tratteggiato: riferimento invariato. "
               "Turchese: previsione che si aggiorna modificando trend e stagionalità. "
               "Variabilità, lead time e altri parametri logistici agiscono sulle scorte, non su questa curva.")
    st.plotly_chart(demand_chart(history, base_forecast, scenario_forecast),
                    use_container_width=True, config={"displaylogo": False})
    table_col, export_col = st.columns([2, 1])
    with table_col:
        st.subheader("Previsioni mensili")
        future_table = pd.DataFrame({
            "Mese": base_forecast.index.strftime("%Y-%m"),
            "Domanda base (unità)": np.round(base_forecast.values, 0).astype(int),
            "Domanda scenario (unità)": np.round(scenario_forecast.values, 0).astype(int),
            "Differenza (unità)": np.round(scenario_forecast.values-base_forecast.values, 0).astype(int),
        })
        st.dataframe(future_table, hide_index=True, use_container_width=True)
        st.download_button("Scarica previsioni (CSV)", csv_content(future_table),
                           "previsioni_scenario.csv", "text/csv")
    with export_col:
        st.subheader("Scheda dell'articolo")
        st.write(f"**Profilo:** {PRODUCTS[selected]['pattern']}")
        st.write(f"**Periodo dei dati simulati:** {history.index[0]:%m/%Y} – {history.index[-1]:%m/%Y} (36 mesi)")
        st.write("**Validazione:** ultimi 12 mesi, previsione un passo avanti")
        st.write("**Orizzonte futuro:** 6 mesi")
        st.write(f"**Volatilità di riferimento (proxy):** {cv*100:.1f}%")
        st.write(f"**Capacità scenario:** {scenario_capacity:,.0f} unità/mese")

    st.subheader("Confronto tra tutti gli articoli — scenario di riferimento")
    summary = []
    for product in PRODUCTS:
        h, report, valid = analyze_product(product, data)
        winner = str(report.loc[0, "Metodo"])
        projection = forecast_series(h, winner, 6)
        summary.append({"Articolo": product, "Profilo": PRODUCTS[product]["pattern"],
                        "Metodo selezionato": winner,
                        "MAE (unità)": round(float(report.loc[0, "MAE"]), 1),
                        "Previsione media/mese": round(float(projection.mean()))})
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

with tab_models:
    st.subheader("Confronto dei cinque metodi")
    st.caption("Per ogni mese di validazione si predice solo il mese successivo usando "
               "esclusivamente le osservazioni disponibili fino al mese precedente. "
               "È un confronto temporale con finestra di addestramento crescente, non casuale.")
    ranking = scores.copy()
    ranking.insert(0, "Posizione", np.arange(1, len(ranking)+1))
    st.dataframe(ranking, hide_index=True, use_container_width=True,
                 column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "sMAPE (%)": st.column_config.NumberColumn(format="%.2f")})
    st.success(f"Metodo selezionato per **{selected}**: **{best}** "
               f"(MAE {scores.loc[0, 'MAE']:.1f} unità; RMSE {scores.loc[0, 'RMSE']:.1f}).")
    col_acc, col_valid = st.columns([1, 1.25])
    with col_acc:
        st.subheader("Errore assoluto medio (MAE)")
        err_fig = go.Figure(go.Bar(
            x=scores["MAE"], y=scores["Metodo"], orientation="h",
            marker_color=[TEAL if i == 0 else INDIGO for i in range(len(scores))],
            text=scores["MAE"].map(lambda x: f"{x:.1f}"), textposition="outside",
        ))
        err_fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(plot_style(err_fig, 370), use_container_width=True,
                        config={"displaylogo": False})
    with col_valid:
        st.subheader("Validazione: previsto rispetto a osservato")
        st.plotly_chart(validation_chart(validation, best), use_container_width=True,
                        config={"displaylogo": False})
    with st.expander("Che cosa misurano MAE, RMSE e sMAPE?"):
        st.markdown("**MAE**: errore assoluto medio espresso in unità; più basso è meglio.  "
                    "**RMSE**: radice dell'errore quadratico medio, più sensibile agli errori grandi.  "
                    "**sMAPE**: errore percentuale assoluto simmetrico; utile al confronto di scale, "
                    "con convenzione zero quando previsione e osservazione sono entrambe nulle.  "
                    "La scelta automatica minimizza MAE e usa RMSE come spareggio.")

with tab_inventory:
    st.subheader("Composizione delle scorte: riferimento vs scenario")
    st.caption("Il riferimento usa: lead time 30 giorni, deviazione 10%, revisione 30 giorni, "
               "servizio 95%, capacità pari al 110% della domanda media base e "
               "previsione non modificata. Lo scenario segue i controlli laterali.")
    st.plotly_chart(inventory_chart(base_stock, scenario_stock),
                    use_container_width=True, config={"displaylogo": False})
    st.dataframe(stock_rows(base_stock, scenario_stock), hide_index=True,
                 use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Scorta fisica media teorica — scenario¹", f"{scenario_stock.physical_reference:,.0f} un.")
    m2.metric("Unità attese in transito", f"{scenario_stock.pipeline:,.0f} un.")
    m3.metric("Livello ordine-fino-a (S)", f"{scenario_stock.order_up_to:,.0f} un.")
    st.caption("¹ Scorta ciclica media + sicurezza + anticipazione; è un indicatore teorico, "
               "non la giacenza istantanea. La pipeline è merce in transito, esclusa "
               "dalla giacenza fisica. Le quattro barre non vanno interpretate come "
               "scorte tutte contemporaneamente presenti nel magazzino.")
    st.subheader("Interpretazione del risultato")
    difference = scenario_stock.physical_reference - base_stock.physical_reference
    st.write(
        f"Con i parametri selezionati la domanda media mensile prevista è **{scenario_stock.avg_monthly_demand:,.0f} unità**; "
        f"la scorta ciclica media è **{scenario_stock.cyclic:,.0f}**, "
        f"quella di sicurezza **{scenario_stock.safety:,.0f}**, "
        f"quella anticipata **{scenario_stock.anticipation:,.0f}** e "
        f"la pipeline **{scenario_stock.pipeline:,.0f}** unità. "
        f"La somma delle componenti fisiche teoriche cambia di **{difference:+,.0f} unità** "
        "rispetto al riferimento. Sono grandezze modellate sotto ipotesi semplificative."
    )
    st.download_button("Scarica confronto scorte (CSV)",
                       csv_content(stock_rows(base_stock, scenario_stock)),
                       "confronto_scorte.csv", "text/csv")

with tab_method:
    st.subheader("Ipotesi, calcoli e limiti")
    st.markdown("""
**Contesto:** impresa reale scelta come riferimento merceologico: Decathlon.
I quattro articoli sono categorie esemplificative, **non codici SKU ufficiali**.
Il punto vendita, i volumi e la domanda non sono riconducibili a dati aziendali reali.

**Dati:** 36 mesi, novembre 2023–ottobre 2026, 4 articoli, 144 record.
Generazione con seme fisso e andamenti distinti: stabilità, stagionalità estiva,
stagionalità invernale, trend crescente. Previsioni novembre 2026–aprile 2027.

**Metodi:** media mobile su 3 mesi, attenuazione esponenziale semplice
(alpha 0,35), regressione lineare, naïve stagionale a 12 mesi e
regressione lineare con indicatori mensili (stagionalità additiva).
La validazione consiste in 12 previsioni rolling a un passo sui 12 mesi finali;
inizialmente sono disponibili 24 mesi. Per ciascuna prova si usano soltanto dati precedenti.

**Scorte — convenzione:** mese commerciale = 30 giorni, con domanda media giornaliera
uguale alla previsione mensile media divisa per 30. Il CV mensile di riferimento
è approssimato come RMSE del metodo migliore / domanda media osservata
nella validazione, limitato tra il 4% e il 60%. È un **indicatore proxy**
dell'incertezza, non una stima rigorosa della deviazione standard della domanda.

- **Ciclica:** domanda giornaliera media × intervallo di riordino / 2.
- **Pipeline:** domanda giornaliera media × lead time medio; stock in transito.
- **Sicurezza:** quantile normale z del livello di servizio × radice quadrata di
  [(lead time + intervallo di riordino) × varianza della domanda giornaliera
  + (domanda giornaliera media × deviazione standard del lead time)²].
- **Anticipata:** massimo tra zero e il più alto fabbisogno cumulato
  dei sei mesi futuri, calcolato come somma di (domanda prevista mensile − capacità
  massima mensile di rifornimento). Si suppone di poter accumulare nei mesi iniziali
  ogni capacità non utilizzata, senza vincoli aggiuntivi.
- **Ordine-fino-a (S):** domanda media durante lead time + intervallo di riordino,
  più scorta di sicurezza e anticipata. È un livello obiettivo modellato, non
  una quantità d'ordine già calcolata su giacenze effettive.

**Scenari:** il trend modifica progressivamente le previsioni a ogni mese futuro;
la stagionalità scala l'ampiezza degli indici mensili storici detrendizzati;
la variabilità moltiplica il CV. Lead time, incertezza sul lead time,
revisione, servizio e capacità sono modificabili indipendentemente.

**Limiti:** ipotesi di domanda e lead time approssimativamente indipendenti,
nessun vincolo su lotti minimi, trasporto, interruzioni di fornitura, prezzi,
costi di mantenimento, resi o stockout reali. La domanda generata non è una
previsione certificata di Decathlon e l'applicazione non gestisce ordini reali.
""")
    st.subheader("Descrizione dei metodi")
    st.dataframe(pd.DataFrame([{"Metodo": k, "Descrizione": v} for k, v in METHODS.items()]),
                 hide_index=True, use_container_width=True)
    st.subheader("Dataset originale simulato")
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.download_button("Scarica dataset (CSV)", csv_content(data),
                       "domanda_decathlon_simulata.csv", "text/csv")

