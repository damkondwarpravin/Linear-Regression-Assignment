import io
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.utils.data_prep import preprocess_dataframe, return_inference

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'linear_regression_model.pkl'
DB_PATH = APP_DIR / 'housing_data.db'

app = FastAPI(title='Linear Regression Inference API')


@app.get('/')
def home():
    return {'message': 'Linear Regression inference endpoint is running.'}


@app.post('/upload-and-preprocess-data')
async def upload_and_preprocess_data(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Please upload a CSV file.')

    contents = await file.read()
    try:
        dataframe = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid CSV file: {exc}') from exc

    processed = preprocess_dataframe(dataframe)
    with __import__('sqlite3').connect(str(DB_PATH)) as conn:
        processed.to_sql('ingested_housing_data', conn, index=False, if_exists='replace')

    return {
        'message': 'Data uploaded, preprocessed and ingested into SQLite.',
        'database_path': str(DB_PATH),
        'rows_loaded': len(processed),
        'columns': list(processed.columns),
        'preview': processed.head(5).to_dict(orient='records'),
    }


@app.post('/inference')
async def inference(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Please upload a CSV file containing raw features.')

    contents = await file.read()
    try:
        raw_data = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid CSV file: {exc}') from exc

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail='No trained model found. Train the model first.')

    prepared = preprocess_dataframe(raw_data)
    if 'price' in prepared.columns:
        prepared = prepared.drop(columns=['price'])

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(prepared).tolist()
    return {'predictions': predictions}


@app.post('/return-inference')
async def return_inference_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Please upload a CSV file containing raw features.')

    contents = await file.read()
    try:
        raw_data = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid CSV file: {exc}') from exc

    predictions = return_inference(MODEL_PATH, raw_data)
    return {'predictions': predictions}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('assignment_2_inference_endpoint:app', host='127.0.0.1', port=8001, reload=False)
