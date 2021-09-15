# Tracing Stock Price - LSTM

## Model
- convert tensorflow 1 to tensorflow 2
- using technical indicators
  - `talib`
  - <https://mrjbq7.github.io/ta-lib/install.html>

## Data

daily

- replace `sklearn.preprocessing.scale` with `sklearn.preprocessing.StandardScaler`
- use `sklearn.preprocessing.OneHotEncoder` with `sklearn.pipeline.Pipeline`

## Tuning
- setup json files
