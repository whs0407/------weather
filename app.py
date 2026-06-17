import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error


st.set_page_config(layout="wide", page_title="기대수명 예측 대시보드")
st.title("🧬 WHO 기대수명 예측 및 모델 성능 비교 웹 서비스")

# -------------------------------------------------------------------------
# 🔥 [오류 수정] @st.set_data를 올바른 캐싱 데코레이터인 @st.cache_resource로 변경
# -------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    m1 = joblib.load('model1_linear.pkl')
    m2 = joblib.load('model2_poly.pkl')
    m3 = joblib.load('model3_ridge.pkl')
    X_train, y_train, X_test, y_test, features = joblib.load('data_splits.pkl')
    return m1, m2, m3, X_train, y_train, X_test, y_test, features

try:
    model1, model2, model3, X_train, y_train, X_test, y_test, features = load_resources()
except FileNotFoundError:
    st.error("모델 파일(.pkl)을 찾을 수 없습니다. 먼저 학습 스크립트를 실행하여 모델을 생성해주세요.")
    st.stop()

# -------------------------------------------------------------------------
# [조건 3] 웹 서비스 내 '각 모델의 성능 비교' 화면 구현 (2단계)
# -------------------------------------------------------------------------
st.header("📊 [2단계] 각 모델의 성능 비교")


def calculate_metrics(model, X_tr, y_tr, X_te, y_te):
    pred_train = model.predict(X_tr)
    pred_test = model.predict(X_te)
    
    r2_train = r2_score(y_tr, pred_train)
    r2_test = r2_score(y_te, pred_test)
    mse_train = mean_squared_error(y_tr, pred_train)
    mse_test = mean_squared_error(y_te, pred_test)
    
   
    if 'poly' in model.named_steps:
        complexity = model.named_steps['poly'].n_output_features_
    else:
        complexity = X_tr.shape[1]
        
    return r2_train, r2_test, mse_train, mse_test, complexity

m1_metrics = calculate_metrics(model1, X_train, y_train, X_test, y_test)
m2_metrics = calculate_metrics(model2, X_train, y_train, X_test, y_test)
m3_metrics = calculate_metrics(model3, X_train, y_train, X_test, y_test)


metrics_data = {
    'Model': ['Model 1 (Linear)', 'Model 2 (Poly)', 'Model 3 (Ridge)'],
    'Train R^2': [m1_metrics[0], m2_metrics[0], m3_metrics[0]],
    'Test R^2': [m1_metrics[1], m2_metrics[1], m3_metrics[1]],
    'Train MSE': [m1_metrics[2], m2_metrics[2], m3_metrics[2]],
    'Test MSE': [m1_metrics[3], m2_metrics[3], m3_metrics[3]],
    'Complexity': [m1_metrics[4], m2_metrics[4], m3_metrics[4]]
}
metrics_df = pd.DataFrame(metrics_data).set_index('Model')


st.subheader("📋 성능 평가 지표 테이블")
st.dataframe(metrics_df, use_container_width=True)


st.subheader("📉 Test R^2 점수 비교 시각화")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=metrics_data['Model'], y=metrics_data['Test R^2'], ax=ax, palette='Set2')
ax.set_title("Comparison of Test R^2 Scores")
ax.set_ylabel("Test R^2 Score")
ax.set_ylim(min(metrics_data['Test R^2']) - 0.1, max(metrics_data['Test R^2']) + 0.1)


for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

st.pyplot(fig)

st.markdown("---")

# -------------------------------------------------------------------------
# [조건 4] 웹 서비스 구현 및 실시간 예측 UI 구성 (3단계)
# -------------------------------------------------------------------------
st.header("🔮 [3단계] 실시간 기대수명 예측 UI")


st.sidebar.header("🛠️ 특성 제어 슬라이더")


input_values = {}
for col in features:
    min_val = float(X_train[col].min())
    max_val = float(X_train[col].max())
    mean_val = float(X_train[col].mean())
    
    input_values[col] = st.sidebar.slider(
        label=f"{col} 범위",
        min_value=min_val * 0.5, 
        max_value=max_val * 1.5,
        value=mean_val
    )


st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox(
    "🤖 분석할 모델을 선택하세요",
    ['Linear', 'Poly', 'Ridge']
)


model_map = {
    'Linear': model1,
    'Poly': model2,
    'Ridge': model3
}
chosen_model = model_map[selected_model_name]


input_df = pd.DataFrame([input_values])
prediction = chosen_model.predict(input_df)[0]


st.subheader(f"🎯 {selected_model_name} 모델의 실시간 예측 결과")
st.markdown(f"<h1 style='text-align: center; color: #1E88E5; font-size: 50px;'>예측된 기대수명: {prediction:.2f} 세</h1>", unsafe_allow_html=True)


st.caption("현재 선택된 슬라이더 입력값:")
st.dataframe(input_df)
