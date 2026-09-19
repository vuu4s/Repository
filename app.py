from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml
from sklearn.metrics import ConfusionMatrixDisplay

from src.dashboard_pipeline import (
    AnalysisBundle,
    fit_baseline_analysis,
    fit_lstm_analysis,
    prepare_dataset,
    run_prediction_backtest,
)
from src.data_loader import generate_mock_ohlcv
from src.labels import ID_TO_LABEL
from src.visualization import plot_equity, plot_indicators, plot_ohlcv, plot_signals
from src.walk_forward import walk_forward_evaluate


ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_config() -> dict[str, object]:
    with (ROOT / "config" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def init_state() -> None:
    defaults = {
        "raw": None,
        "bundle": None,
        "baseline": None,
        "lstm": None,
        "backtests": {},
        "walk_forward": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_raw_data(uploaded_file, symbol: str, config: dict[str, object]) -> pd.DataFrame:
    if uploaded_file is None:
        return generate_mock_ohlcv(config["data"]["mock_rows"], config["project"]["random_seed"])
    frame = pd.read_csv(uploaded_file, parse_dates=["datetime"])
    if "datetime" not in frame.columns:
        raise ValueError("CSV 必須包含 datetime 欄位")
    frame = frame.set_index("datetime").sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位: {sorted(missing)}")
    return frame[sorted(required)]


def ensure_bundle(raw: pd.DataFrame, controls: dict[str, float | int], config: dict[str, object]) -> AnalysisBundle:
    return prepare_dataset(
        raw,
        config["features"],
        horizon=int(controls["horizon"]),
        buy_threshold=float(controls["buy_threshold"]),
        sell_threshold=float(controls["sell_threshold"]),
        train_ratio=float(controls["train_ratio"]),
        validation_ratio=float(controls["validation_ratio"]),
    )


def costs_from_controls(controls: dict[str, float | int]) -> dict[str, float]:
    return {
        "initial_cash": 1_000_000.0,
        "fee_rate": float(controls["fee_rate"]),
        "tax_rate": float(controls["tax_rate"]),
        "slippage_rate": float(controls["slippage_rate"]),
        "position_size": 1.0,
    }


def train_lstm(bundle: AnalysisBundle, config: dict[str, object], controls: dict[str, float | int]) -> dict[str, object]:
    return fit_lstm_analysis(
        bundle,
        sequence_length=int(config["split"]["sequence_length"]),
        hidden_size=int(config["models"]["lstm_hidden_size"]),
        num_layers=int(config["models"]["lstm_layers"]),
        epochs=int(config["models"]["epochs"]),
        learning_rate=float(config["models"]["learning_rate"]),
        batch_size=int(config["models"]["batch_size"]),
        random_seed=int(config["project"]["random_seed"]),
    )


def run_backtests(bundle: AnalysisBundle, baseline: dict[str, object] | None, lstm: dict[str, object] | None, controls: dict[str, float | int]) -> dict[str, object]:
    results: dict[str, object] = {}
    costs = costs_from_controls(controls)
    if baseline is not None:
        for name, evaluation in baseline["evaluations"].items():
            result, summary = run_prediction_backtest(
                bundle.split.test["close"],
                evaluation["predictions"],
                bundle.split.test.index,
                **costs,
            )
            results[name] = {"result": result, "summary": summary}
    if lstm is not None:
        result, summary = run_prediction_backtest(
            bundle.split.test["close"],
            lstm["predictions"],
            lstm["test_index"],
            **costs,
        )
        results["lstm"] = {"result": result, "summary": summary}
    return results


def render_metrics(metrics: dict[str, float]) -> None:
    columns = st.columns(5)
    for column, (name, value) in zip(columns, metrics.items()):
        shown = "N/A" if pd.isna(value) else f"{value:.4f}"
        column.metric(name.upper(), shown)


def render_data(bundle: AnalysisBundle) -> None:
    st.subheader("資料概況")
    columns = st.columns(4)
    columns[0].metric("資料開始日期", str(bundle.raw.index.min()))
    columns[1].metric("資料結束日期", str(bundle.raw.index.max()))
    columns[2].metric("K棒數量", f"{len(bundle.raw):,}")
    columns[3].metric("可分析筆數", f"{len(bundle.frame):,}")
    st.dataframe(bundle.raw.tail(300), use_container_width=True)
    st.pyplot(plot_ohlcv(bundle.raw), clear_figure=True)


def render_ml(bundle: AnalysisBundle, baseline: dict[str, object] | None) -> None:
    if baseline is None:
        st.info("請先按「訓練 ML」或「完整分析」。")
        return
    rows = []
    for name, evaluation in baseline["evaluations"].items():
        rows.append({"模型": name.replace("_", " ").title(), **evaluation["metrics"]})
    st.dataframe(pd.DataFrame(rows).set_index("模型"), use_container_width=True)
    selected = st.selectbox("查看 confusion matrix", list(baseline["evaluations"]))
    evaluation = baseline["evaluations"][selected]
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        bundle.split.test["label"].map({"SELL": 0, "HOLD": 1, "BUY": 2}),
        evaluation["predictions"],
        labels=[0, 1, 2],
        display_labels=["SELL", "HOLD", "BUY"],
        cmap="Blues",
        ax=axis,
    )
    st.pyplot(figure, clear_figure=True)


def render_lstm(lstm: dict[str, object] | None) -> None:
    if lstm is None:
        st.info("請先按「訓練 LSTM」或「完整分析」。")
        return
    history = pd.DataFrame(lstm["history"])
    st.line_chart(history)
    render_metrics(lstm["metrics"])
    probability = lstm["probabilities"][-1]
    st.dataframe(
        pd.DataFrame(
            {"類別": [ID_TO_LABEL[index] for index in range(3)], "機率": probability}
        ).set_index("類別"),
        use_container_width=True,
    )


def render_signals(bundle: AnalysisBundle, baseline: dict[str, object] | None, lstm: dict[str, object] | None) -> None:
    if lstm is not None:
        probability = lstm["probabilities"][-1]
        signal = ID_TO_LABEL[int(lstm["predictions"][-1])]
    elif baseline is not None:
        evaluation = baseline["evaluations"]["random_forest"]
        probability = evaluation["probabilities"][-1]
        signal = ID_TO_LABEL[int(evaluation["predictions"][-1])]
    else:
        st.info("請先訓練模型。")
        return
    st.warning("AI 訊號僅供模型研究與歷史分析，不是保證獲利或投資建議。")
    st.metric("目前訊號", signal)
    columns = st.columns(3)
    label_to_index = {"BUY": 2, "HOLD": 1, "SELL": 0}
    for index, label in enumerate(["BUY", "HOLD", "SELL"]):
        columns[index].metric(f"{label} 機率", f"{probability[label_to_index[label]]:.2%}")


def render_backtest(backtests: dict[str, object]) -> None:
    if not backtests:
        st.info("請先按「執行回測」或「完整分析」。")
        return
    selected = st.selectbox("回測模型", list(backtests), key="backtest_model")
    item = backtests[selected]
    summary = item["summary"]
    render_metrics({
        "total trades": summary["total_trades"],
        "win rate": summary["win_rate"],
        "strategy return": summary["strategy_return"],
        "average trade": summary["average_trade_return"],
        "profit factor": summary["profit_factor"],
    })
    st.dataframe(pd.DataFrame([summary]), use_container_width=True)
    st.pyplot(plot_signals(item["result"]), clear_figure=True)
    st.pyplot(plot_equity(item["result"]), clear_figure=True)


def main() -> None:
    st.set_page_config(page_title="台股當沖 AI 分析系統", layout="wide")
    init_state()
    config = load_config()
    st.title("台股當沖 AI 分析系統")
    st.caption("ML / LSTM 智慧學習與歷史回測")

    data_config = config["data"]
    model_config = config["models"]
    split_config = config["split"]
    backtest_config = config["backtest"]
    with st.sidebar:
        st.header("分析設定")
        symbol = st.text_input("股票代號", value=data_config["symbol"])
        st.selectbox("K棒週期", ["1min", "5min", "15min"], index=0)
        uploaded_file = st.file_uploader("資料檔案", type=["csv"])
        horizon = st.number_input("預測未來 N 根K", min_value=1, max_value=120, value=int(data_config["horizon_bars"]))
        buy_threshold = st.number_input("BUY threshold", min_value=0.0, value=float(data_config["label_threshold"]), format="%.4f")
        sell_threshold = st.number_input("SELL threshold", min_value=0.0, value=float(data_config["label_threshold"]), format="%.4f")
        train_ratio = st.slider("訓練比例", 0.1, 0.8, float(split_config["train_ratio"]), 0.05)
        validation_ratio = st.slider("Validation 比例", 0.05, 0.4, float(split_config["validation_ratio"]), 0.05)
        test_ratio = st.slider("Test 比例", 0.05, 0.4, float(split_config["test_ratio"]), 0.05)
        fee_rate = st.number_input("手續費", min_value=0.0, value=float(backtest_config["fee_rate"]), format="%.6f")
        tax_rate = st.number_input("證交稅", min_value=0.0, value=float(backtest_config["tax_rate"]), format="%.6f")
        slippage_rate = st.number_input("滑價", min_value=0.0, value=float(backtest_config["slippage_rate"]), format="%.6f")
        controls = locals()
        controls = {key: controls[key] for key in ["horizon", "buy_threshold", "sell_threshold", "train_ratio", "validation_ratio", "test_ratio", "fee_rate", "tax_rate", "slippage_rate"]}
        st.caption("資料切分採 chronological split；Scaler 僅 fit training data。")
        load_clicked = st.button("載入資料", use_container_width=True)
        features_clicked = st.button("開始特徵工程", use_container_width=True)
        ml_clicked = st.button("訓練 ML", use_container_width=True)
        lstm_clicked = st.button("訓練 LSTM", use_container_width=True)
        backtest_clicked = st.button("執行回測", use_container_width=True)
        walk_clicked = st.button("執行 Walk-forward", use_container_width=True)
        full_clicked = st.button("完整分析", type="primary", use_container_width=True)

    try:
        if load_clicked or full_clicked:
            st.session_state.raw = load_raw_data(uploaded_file, symbol, config)
            st.session_state.bundle = None
            st.session_state.baseline = None
            st.session_state.lstm = None
            st.session_state.backtests = {}
            st.session_state.walk_forward = None
        if st.session_state.raw is not None and (features_clicked or full_clicked or st.session_state.bundle is None and (ml_clicked or lstm_clicked or backtest_clicked or walk_clicked)):
            if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("訓練、Validation、Test 比例總和必須等於 1")
            st.session_state.bundle = ensure_bundle(st.session_state.raw, controls, config)
        if ml_clicked or full_clicked:
            st.session_state.baseline = fit_baseline_analysis(
                st.session_state.bundle,
                n_estimators=int(model_config["random_forest_estimators"]),
                random_state=int(config["project"]["random_seed"]),
            )
        if lstm_clicked or full_clicked:
            st.session_state.lstm = train_lstm(st.session_state.bundle, config, controls)
        if backtest_clicked or full_clicked:
            st.session_state.backtests = run_backtests(
                st.session_state.bundle,
                st.session_state.baseline,
                st.session_state.lstm,
                controls,
            )
        if walk_clicked or full_clicked:
            st.session_state.walk_forward = walk_forward_evaluate(
                st.session_state.bundle.frame,
                st.session_state.bundle.feature_names,
                train_ratio=float(controls["train_ratio"]),
                validation_ratio=float(controls["validation_ratio"]),
                test_ratio=float(controls["test_ratio"]),
                n_estimators=int(model_config["random_forest_estimators"]),
                random_state=int(config["project"]["random_seed"]),
            )
    except Exception as error:
        st.error(f"分析失敗：{error}")

    bundle = st.session_state.bundle
    if bundle is None:
        st.info("請從左側按「載入資料」開始。預設使用可重現的 mock 分鐘資料。")
        return
    data_tab, feature_tab, ml_tab, lstm_tab, signal_tab, backtest_tab, compare_tab = st.tabs(
        ["資料", "技術指標", "ML 模型", "LSTM", "AI 訊號", "回測", "模型比較"]
    )
    with data_tab:
        render_data(bundle)
    with feature_tab:
        st.pyplot(plot_indicators(bundle.frame), clear_figure=True)
        st.dataframe(bundle.frame.tail(300), use_container_width=True)
    with ml_tab:
        render_ml(bundle, st.session_state.baseline)
    with lstm_tab:
        render_lstm(st.session_state.lstm)
    with signal_tab:
        render_signals(bundle, st.session_state.baseline, st.session_state.lstm)
    with backtest_tab:
        render_backtest(st.session_state.backtests)
    with compare_tab:
        comparison = []
        for name, evaluation in (st.session_state.baseline or {}).get("evaluations", {}).items():
            metrics = evaluation["metrics"]
            strategy = st.session_state.backtests.get(name, {}).get("summary", {})
            comparison.append({"模型": name, **metrics, **strategy})
        if st.session_state.lstm is not None:
            metrics = st.session_state.lstm["metrics"]
            strategy = st.session_state.backtests.get("lstm", {}).get("summary", {})
            comparison.append({"模型": "LSTM", **metrics, **strategy})
        if comparison:
            st.dataframe(pd.DataFrame(comparison).set_index("模型"), use_container_width=True)
        else:
            st.info("請先訓練模型並執行回測。")
        if st.session_state.walk_forward is not None:
            st.subheader("Walk-forward evaluation")
            st.dataframe(st.session_state.walk_forward, use_container_width=True)


if __name__ == "__main__":
    main()
