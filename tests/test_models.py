import numpy as np
import pandas as pd
import pytest

from src.dataset import PRODUCTS, generate_dataset, load_dataset
from src.forecast import METHODS, evaluate_methods, forecast_series, error_metrics
from src.scenarios import transform_forecast
from src.inventory import calculate_inventory


def product_history(df, name):
    sub = df[df['articolo'] == name].sort_values('mese')
    return pd.Series(sub['domanda_unita'].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(sub['mese']))


def test_dataset_is_reproducible_and_valid(tmp_path):
    a, b = generate_dataset(), generate_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (144, 4)
    assert set(a.articolo) == set(PRODUCTS)
    assert set(a.natura_dato) == {'SIMULATO'}
    path = tmp_path / 'test.csv'
    a.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.shape == (144, 4)
    assert not loaded.duplicated(['mese', 'articolo']).any()


@pytest.mark.parametrize('product', list(PRODUCTS))
def test_methods_validate_and_forecast_each_product(product):
    data = generate_dataset()
    history = product_history(data.assign(mese=pd.to_datetime(data.mese)), product)
    ranking, rolling_predictions = evaluate_methods(history, holdout=12)
    assert len(ranking) == 5
    assert len(rolling_predictions) == 12
    assert ranking.MAE.is_monotonic_increasing
    assert ranking['MAE'].ge(0).all()
    assert ranking['RMSE'].ge(0).all()
    assert np.isfinite(ranking[['MAE','RMSE','sMAPE (%)']].to_numpy()).all()
    for method in METHODS:
        forecast = forecast_series(history, method, 6)
        assert len(forecast) == 6
        assert forecast.index[0] == pd.Timestamp('2026-11-01')
        assert forecast.index[-1] == pd.Timestamp('2027-04-01')
        assert (forecast >= 0).all()
        # Earliest validation forecast sees only 24 historical months.
        single_step = forecast_series(history.iloc[:24], method, 1).iloc[0]
        assert rolling_predictions.iloc[0][method] == pytest.approx(single_step)


def test_validation_is_causally_ordered():
    data = generate_dataset()
    h = product_history(data.assign(mese=pd.to_datetime(data.mese)), 'Tappetini fitness')
    _, baseline = evaluate_methods(h, 12)
    altered = h.copy()
    altered.iloc[-1] += 8000
    _, changed = evaluate_methods(altered, 12)
    # Future data cannot affect the previous validation months' predictions.
    pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1])
    # Even the last prediction is made before observing the changed last month.
    pd.testing.assert_series_equal(baseline.iloc[-1, 1:], changed.iloc[-1, 1:], check_names=False)


def test_scenario_identity_and_trend():
    data = generate_dataset()
    h = product_history(data.assign(mese=pd.to_datetime(data.mese)), 'Giacche termiche')
    base = forecast_series(h, 'Regressione trend + stagionalità', 6)
    default = transform_forecast(base, h, 0, 100)
    np.testing.assert_allclose(default, base)
    increased = transform_forecast(base, h, 3, 100)
    assert np.all(increased > base)
    changed_season = transform_forecast(base, h, 0, 0)
    assert not np.allclose(changed_season, base)


def test_stock_stays_nonnegative_and_responds_to_parameters():
    forecast = pd.Series([200, 250, 400, 500, 300, 230],
                         index=pd.date_range('2026-11-01', periods=6, freq='MS'))
    base = calculate_inventory(forecast, .1, 30, 3, 30, .95, 300)
    varied = calculate_inventory(forecast, .3, 60, 6, 30, .95, 300)
    assert base.cyclic > 0
    assert base.safety > 0
    assert base.anticipation >= 0
    assert base.pipeline > 0
    assert varied.safety > base.safety
    assert varied.pipeline > base.pipeline
    assert base.physical_reference == pytest.approx(base.cyclic + base.safety + base.anticipation)
    assert base.composite_reference == pytest.approx(base.physical_reference + base.pipeline)
    assert base.order_up_to > base.safety


def test_anticipation_is_capacity_sensitive():
    f = pd.Series([100, 120, 400, 500, 210, 180])
    high = calculate_inventory(f, .1, 30, 0, 30, .95, 600)
    low = calculate_inventory(f, .1, 30, 0, 30, .95, 150)
    assert high.anticipation == 0
    assert low.anticipation > high.anticipation


def test_no_zero_division_in_smape():
    metrics = error_metrics(np.array([0.0, 4.0]), np.array([0.0, 5.0]))
    assert np.isfinite(list(metrics.values())).all()
    assert 0 <= metrics['sMAPE (%)'] <= 200


def test_invalid_input_fails():
    with pytest.raises(ValueError):
        calculate_inventory(pd.Series([10.]), .1, 30, 2, 30, 1.0, 100)
