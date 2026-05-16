# 零报错 部署界面代码
import os
import re
import gradio as gr
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "tfidf.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"缺少模型文件。请先运行 train.py 生成模型文件：{MODEL_PATH} 和 {VECTORIZER_PATH}"
    )

# 加载模型
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# 文本清洗函数
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    stop_words = set([w.strip() for w in [
        'a','about','above','after','again','against','all','am','an','and','any','are','aren','as','at',
        'be','because','been','before','being','below','between','both','but','by','can','could','couldn',
        'did','didn','do','does','doesn','doing','don','down','during','each','few','for','from','further',
        'had','hadn','has','hasn','have','haven','having','he','her','here','hers','herself','him','himself',
        'his','how','i','if','in','into','is','isn','it','its','itself','let','me','more','most','mustn','my',
        'myself','no','nor','not','of','off','on','once','only','or','other','our','ours','ourselves','out',
        'over','own','same','shan','she','should','shouldn','so','some','such','than','that','the','their',
        'theirs','them','themselves','then','there','these','they','this','those','through','to','too',
        'under','until','up','very','was','wasn','we','were','weren','what','when','where','which','while',
        'who','whom','why','will','with','won','would','wouldn','you','your','yours','yourself','yourselves'
    ]])
    words = [w for w in words if w and w not in stop_words]
    return ' '.join(words)

def predict_sentiment(text):
    if not text:
        return "请输入评论"
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    result = model.predict(vec)[0]
    return "正面评价" if result == 1 else "负面评价"

# 界面
with gr.Blocks() as demo:
    gr.Markdown("# IMDB 电影评论情感分析模型")
    input_text = gr.Textbox(label="输入英文电影评论")
    btn = gr.Button("开始分析")
    output_label = gr.Label(label="情感结果")
    btn.click(predict_sentiment, inputs=input_text, outputs=output_label)

# 启动
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)