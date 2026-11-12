import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعداد الصفحة
st.set_page_config(page_title="Credit Default Prediction", layout="wide")
st.title("💸 Credit Default Risk Prediction")
st.write("قم برفع ملف CSV يحتوي على بيانات العملاء لتوقع احتمالية تعثرهم في السداد.")

# تحميل النموذج من مجلد models
@st.cache_resource
def load_model():
    # المسار معدل ليقرأ النموذج من فولدر models
    return joblib.load('models/voting_pipeline_model.pkl')

model = load_model()

# دوال تجهيز البيانات
def drop_unnecessary_cols(df):
    cols_to_drop = ['SK_ID_CURR', 'FLAG_MOBIL', 'AMT_GOODS_PRICE']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=existing_cols)

def handle_missing_values(df):
    bureau_cols = [col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
    for col in bureau_cols:
        df[col] = df[col].fillna(0)
    if 'OWN_CAR_AGE' in df.columns:
        df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(-1)
    if 'CNT_FAM_MEMBERS' in df.columns:
        mode_val = 2.0 
        df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(mode_val)
    return df

def handle_anomalies(df):
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'] = df['CODE_GENDER'].replace({'XNA': np.nan})
    return df

def engineer_features(df):
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365
        df = df.drop(columns=['DAYS_BIRTH'])
    if 'DAYS_EMPLOYED' in df.columns:
        df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365
        df = df.drop(columns=['DAYS_EMPLOYED'])
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
        df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    return df

# أداة رفع الملفات
uploaded_file = st.file_uploader("Upload CSV Test Data", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_input.head())
    
    if st.button("Predict Risk"):
        with st.spinner("Processing and predicting..."):
            df_processed = drop_unnecessary_cols(df_input)
            df_processed = handle_missing_values(df_processed)
            df_processed = handle_anomalies(df_processed)
            df_features = engineer_features(df_processed)
            
            probs = model.predict_proba(df_features)[:, 1]
            custom_threshold = 0.25
            preds = (probs >= custom_threshold).astype(int)
            
            results = pd.DataFrame({
                'Risk Probability': np.round(probs, 4),
                'Prediction (1=Default, 0=Repaid)': preds
            })
            
            if 'SK_ID_CURR' in df_input.columns:
                results.insert(0, 'Client ID', df_input['SK_ID_CURR'])
                
            st.write("### Prediction Results")
            st.dataframe(results)
            
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
