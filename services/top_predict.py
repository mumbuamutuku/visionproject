import base64
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def get_top_predictions(pred, class_names, top=3):
    top_indices = pred[0].argsort()[-top:][::-1]
    top_labels = [class_names[i] for i in top_indices]
    top_scores = [float(pred[0][i]) * 100 for i in top_indices]
    return top_labels, top_scores

def plot_prediction_chart(labels, scores):
    if not labels or not scores:
        return ""  
    plt.figure(figsize=(6, 4))
    bars = plt.barh(labels[::-1], scores[::-1], color='skyblue')
    plt.xlabel('Confidence (%)')
    plt.title('Top Predictions')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')