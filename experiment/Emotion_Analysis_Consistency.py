import pandas as pd
from transformers import pipeline

# 데이터 로드
file_path = 'predictions.csv'  # 예시 경로, 실제 파일 경로로 수정해야 합니다.
df = pd.read_csv(file_path)

# 감정 분석 파이프라인 설정
sentiment_pipeline = pipeline("sentiment-analysis")

# 감정 분석 함수 정의
def analyze_sentiments(texts):
    results = sentiment_pipeline(texts)
    return [result['label'] for result in results]

# 감정 분석 실행
df['Generated Sentiment'] = df['Generated Text'].apply(lambda x: analyze_sentiments([x])[0])
df['Actual Sentiment'] = df['Actual Text'].apply(lambda x: analyze_sentiments([x])[0])

# 일치도 평가
def evaluate_match(df):
    match_count = (df['Generated Sentiment'] == df['Actual Sentiment']).sum()
    total_count = len(df)
    match_rate = match_count / total_count
    return match_rate

match_rate = evaluate_match(df)
print(f"감정 일치도: {match_rate:.2%}")

# 선택적: 결과 확인
print(df[['Generated Sentiment', 'Actual Sentiment']].head())
