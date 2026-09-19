# 台股股票當沖 ML/LSTM 學習分析系統

第一階段提供一個不連接券商、不自動下單、只使用 mock 分鐘 K 資料的最小可運作研究流程。

## 功能

- 產生可重現的台股分鐘 OHLCV mock 資料，也可載入 CSV
- MA、EMA、RSI、MACD、ATR、波動率與成交量特徵
- 預測未來 N 根 K 棒的 BUY、HOLD、SELL 三分類
- 依時間順序切分資料，禁止 random split 與 look-ahead bias
- StandardScaler 只在 training data fit
- Logistic Regression、Random Forest 與 PyTorch LSTM
- Walk-forward 評估流程的基礎元件與當沖歷史回測
- 手續費、證交稅、滑價、Accuracy、Precision、Recall、F1、AUC、Strategy Return、Win Rate、Profit Factor、Maximum Drawdown、Sharpe Ratio

## 專案結構

```text
config/config.yaml       主要參數
src/                     資料、特徵、模型、回測與指標
notebooks/               五個繁體中文教學 notebook
tests/                   核心行為測試
requirements.txt         Python 依賴
```

## 使用方式

```bash
python -m pip install -r requirements.txt
pytest -q
```

在專案根目錄開啟 Jupyter notebook，依序執行 `notebooks/01_資料處理.ipynb` 至 `notebooks/05_當沖回測.ipynb`。目前 notebook 使用 mock 資料，不使用真實資金，也不會連接券商或送出交易。

## Streamlit AI 分析控制台

安裝依賴後，在專案根目錄啟動網頁控制台：

```bash
streamlit run app.py
```

控制台可載入 mock 或 CSV 分鐘資料，依序執行特徵工程、ML baseline、CPU PyTorch LSTM、歷史回測與 walk-forward evaluation。Sidebar 可調整股票代號、K 棒週期、預測 horizon、BUY/SELL threshold、時間切分比例與交易成本；「完整分析」會一次執行整個研究流程。

CSV 必須包含 `datetime`、`open`、`high`、`low`、`close`、`volume` 欄位，`datetime` 會作為排序後的時間索引。特徵只使用當下與過去資料，資料採 chronological train/validation/test split，scaler 只在 training fit，validation/test 僅 transform。

控制台比較 Logistic Regression、Random Forest 與 LSTM 的 Accuracy、Precision、Recall、F1、AUC，以及策略報酬、勝率、Profit Factor、Maximum Drawdown 與 Sharpe Ratio。回測只允許空手／多單，成本包含手續費、證交稅與滑價；BUY/HOLD/SELL mapping 在訓練、預測、圖表與回測中一致。

所有 AI 訊號、模型機率與回測結果僅供研究分析，不構成投資建議。此專案不連接券商、不使用 API key、不自動下單，也不執行真實交易。

## 資料格式

真實 CSV 預期包含 `datetime`、`open`、`high`、`low`、`close`、`volume` 欄位，且資料應已依時間排序。所有模型與回測結果僅供研究與測試，不構成投資建議。