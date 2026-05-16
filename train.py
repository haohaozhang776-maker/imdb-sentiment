# 零报错 IMDB 情感分析神经网络训练代码
import os
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "imdb_balanced_10k.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "tfidf.pkl")

# 下载停用词（不报错版）
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

# 文本清洗函数
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words)

# 主程序
if __name__ == "__main__":
    # 1. 读取数据
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    if os.path.getsize(DATA_PATH) == 0:
        raise ValueError(f"数据文件为空: {DATA_PATH}。请确认 imdb_balanced_10k.csv 是否已正确下载或复制。")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError(f"读取到空数据集：{DATA_PATH}")
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("数据文件必须包含 'review' 和 'sentiment' 两列。")
    
    # 2. 清洗评论
    df["cleaned"] = df["review"].apply(clean_text)
    
    # 3. 划分训练集测试集
    X = df["cleaned"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. TF-IDF 向量化
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    
    # 5. 神经网络模型
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=10,
        random_state=42,
        verbose=True
    )
    
    # 6. 训练
    print("开始训练模型...")
    model.fit(X_train_tfidf, y_train)
    
    # 7. 测试
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {acc:.2f}")
    
    # 8. 保存模型
    joblib.dump(model, "model.pkl")
    joblib.dump(tfidf, "tfidf.pkl")
    print("模型保存成功！")