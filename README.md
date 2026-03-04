💰 Fake Currency Detection using Deep Learning

A Deep Learning based Fake Currency Detection System that identifies whether a currency note is REAL or FAKE using MobileNetV2 and TensorFlow.
The system is deployed with an interactive Streamlit web interface where users can upload an image of a currency note and get instant predictions.

📌 Project Overview

Counterfeit currency is a major issue affecting economies worldwide. This project uses Computer Vision and Deep Learning to automatically detect fake currency notes from images.

The model is trained using MobileNetV2 (Transfer Learning) to classify currency notes into two categories:

Real

Fake

The trained model is integrated with a Streamlit web application for easy use.

🚀 Features

Deep Learning based currency detection

MobileNetV2 Transfer Learning model

Image data augmentation for better accuracy

Confusion matrix and classification report generation

Streamlit web interface for easy interaction

Upload currency images for prediction

Clean and modern UI

🧠 Technologies Used

Python

TensorFlow / Keras

MobileNetV2

NumPy

Scikit-learn

Matplotlib

Streamlit

Pillow

📂 Project Structure
validate-indian-currency/
│
├── dataset/
│   ├── training/
│   │   ├── real/
│   │   └── fake/
│   │
│   ├── validation/
│   │   ├── real/
│   │   └── fake/
│   │
│   └── testing/
│       ├── real/
│       └── fake/
│
├── model/
│   └── currency_mobilenet_model.h5
│
├── reports/
│   └── confusion_matrix.png
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation
1️⃣ Clone the repository
git clone (https://github.com/Umesh-BN-07/validate-indian-currency.git)
cd validate-indian-currency
2️⃣ Install dependencies
pip install -r requirements.txt
🏋️ Train the Model

Run the training script:

python train.py

The trained model will be saved in:

model/currency_mobilenet_model.h5

The confusion matrix will be saved in:

reports/confusion_matrix.png
▶️ Run the Web Application

Start the Streamlit application:

streamlit run app.py

Then open the browser at:

http://localhost:8501
🖼️ How It Works

User uploads an image of a currency note.

The image is resized and preprocessed.

The trained MobileNetV2 model analyzes the note.

The model predicts whether the note is REAL or FAKE.

The result is displayed on the web interface.

📊 Model Details
Component	Description
Model	MobileNetV2
Input Size	224 × 224
Batch Size	16
Epochs	12 + Fine Tuning
Loss Function	Binary Crossentropy
Optimizer	Adam
📈 Evaluation Metrics

The model evaluation includes:

Accuracy

Confusion Matrix

Precision

Recall

F1 Score

Example Output:

Test Accuracy: 95%
📷 Example Workflow

1️⃣ Upload currency note image
2️⃣ Click Predict Note Authenticity
3️⃣ System analyzes the image
4️⃣ Result displayed as:

✔ REAL NOTE
or
❌ FAKE NOTE DETECTED
🔮 Future Improvements

Support for multiple currency denominations

Mobile application integration

Real-time camera detection

Larger dataset for improved accuracy

Deployment on cloud platforms

👨‍💻 Author

Umesh

Presented by SJCIT

📜 License

This project is for educational and research purposes only.
