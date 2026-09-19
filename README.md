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

## 資料格式

真實 CSV 預期包含 `datetime`、`open`、`high`、`low`、`close`、`volume` 欄位，且資料應已依時間排序。所有模型與回測結果僅供研究與測試，不構成投資建議。