SMS Spam Detection using TensorFlow
📌 Overview
A TensorFlow-based deep learning solution for classifying SMS messages as spam or ham (legitimate messages). Implements a neural network with text embedding layers for accurate spam detection.

✨ Key Features
Deep Learning Approach: Uses TensorFlow/Keras with embedding layers

Text Preprocessing: Handles raw SMS data cleaning and normalization

Model Evaluation: Comprehensive metrics (accuracy, precision, recall)

Visualizations: Confusion matrix and performance charts

Comparisons: Benchmarked against traditional ML (Naive Bayes)

🛠️ Technical Implementation
Framework: TensorFlow 2.x

Model Architecture:

Text Vectorization Layer

Embedding Layer (128 dimensions)

GlobalAveragePooling1D

Dense Layers with ReLU/Sigmoid activations

Training:

Optimizer: Adam

Loss: BinaryCrossentropy

Metrics: Accuracy

📊 Results
Neural Network Performance:

Accuracy: ~96% (comparable to Naive Bayes)

Precision/Recall metrics for both classes

Confusion matrix visualization

Advantages over Traditional ML:

Better handles complex text patterns

More adaptable to new spam patterns

End-to-end text processing

🚀 How to Use
Clone repo: git clone [https://github.com/shravani2911/Spam-sms-detection/tree/main]

Install requirements: pip install -r requirements.txt

Run notebook: jupyter notebook spam_detection.ipynb

🔮 Future Improvements
Implement LSTM/GRU layers for sequence analysis

Add BERT/Transformer-based approaches

Create Flask API for real-time predictions

Expand to multi-language support

📜 License
MIT Licensed - Free for academic and commercial use
