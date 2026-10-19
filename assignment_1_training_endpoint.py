import io
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.data_prep import preprocess_dataframe, split_data, ingest_to_sqlite

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / 'housing_data.db'
MODEL_PATH = APP_DIR / 'linear_regression_model.pkl'

app = FastAPI(title='Linear Regression Training API')


@app.get('/')
def home():
    return {'message': 'Linear Regression training endpoint is running.'}


@app.post('/train-model')
async def train_model(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Please upload a CSV file.')

    contents = await file.read()
    try:
        dataframe = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid CSV file: {exc}') from exc

    if 'price' not in dataframe.columns:
        raise HTTPException(status_code=400, detail='The uploaded CSV must contain a price column.')

    ingest_to_sqlite(DB_PATH, dataframe, table_name='housing_training_data')

    X_train, X_valid, y_train, y_valid = split_data(dataframe)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    rmse = mean_squared_error(y_valid, predictions, squared=False)
    r2 = r2_score(y_valid, predictions)
    joblib.dump(model, MODEL_PATH)

    return {
        'message': 'Model trained successfully.',
        'database_path': str(DB_PATH),
        'model_path': str(MODEL_PATH),
        'rows_loaded': len(dataframe),
        'training_rows': len(X_train),
        'validation_rows': len(X_valid),
        'rmse': round(float(rmse), 4),
        'r2_score': round(float(r2), 4),
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('assignment_1_training_endpoint:app', host='127.0.0.1', port=8000, reload=False)
