# **Sentiment Analysis with Continuous Performance Monitoring and Local Deployment**

## **Overview**

This project is a comprehensive **Sentiment Analysis Pipeline** built to classify text data and monitor the model’s performance using **MLflow**. The model is deployed locally using **Flask**, allowing for both **batch and live inference requests**. Additionally, the system automatically retrains the model if performance degrades or it becomes stale, ensuring up-to-date and accurate predictions.

Key features include:
- **Model Training and Evaluation:** Efficient training using transformer-based architecture with continuous performance monitoring via **MLflow** and **Wandb.ai**.
- **Performance Monitoring & Staleness Checks:** Automatic detection of model staleness using **MLflow** and **wandb.ai**, triggering model retraining when performance deteriorates.
- **Inference Pipeline:** Supports both **batch** and **live inference** requests using predefined APIs.
- **Alert System:** Email alerts via Gmail are sent when the model's performance falls below a set threshold or becomes stale.
- **Checkpoints:** Integrated checkpoints monitor model performance continuously, ensuring ongoing monitoring and triggering necessary actions like retraining.
- **Local Deployment via Flask:** The model is deployed locally using **Flask**, allowing real-time inference through REST APIs.


---

## **Architecture**

The project follows a multi-stage architecture:

1. **Data Preprocessing:** Tokenization and augmentation of customer reviews.
2. **Model Training:** A transformer-based model is trained for sentiment classification, with metrics logged via **Wandb.ai** and **MLflow**.
3. **Performance Monitoring with MLflow and Wandb.ai:** Logs metrics (accuracy, loss, and more) continuously. **MLflow** also checks for staleness by comparing the latest model's performance with historical metrics.
4. **Checkpoints:** Performance checkpoints are integrated within the training pipeline to ensure continuous monitoring. Alerts or retraining mechanisms trigger based on performance trends over time.
5. **Model Deployment via Flask:** 
   - Batch Inference: Process large batches of reviews in a single request.
   - Live Inference: Real-time review processing through a REST API.
6. **Alerts and Retraining:** If the model becomes stale or underperforms, an alert is triggered, and the model is automatically retrained.

---

## **Dataset**

The dataset used for training the sentiment analysis model is stored at the following path:

```bash
C:\Users\aasho\OneDrive\Desktop\transformer pipeline\data\sentiment_analysis.csv
```

### Dataset Details:
- **Columns:** 
  - `Review`: The text of customer reviews.
  - `Sentiment`: The sentiment labels, either "positive" or "negative".

---

## **Installation**

### Prerequisites:
To run this project, ensure you have the following packages installed:

- Python 3.7+
- Transformers
- Wandb (Weights and Biases)
- MLflow
- Flask
- Pandas
- Scikit-learn
- TensorFlow or PyTorch
- Matplotlib
- Gmail SMTP for alerts

### Step-by-Step Setup:

1. Clone the repository:
   ```bash
   git clone <repository_link>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up **Wandb.ai**:
   ```bash
   wandb login
   ```

4. Set up **Wandd ai** for model tracking:
    
5. Set up Flask for local deployment:
   ```bash
     pip install flask
   ```

6. Add your Gmail credentials for email alerts:
   Update `email_alert.py` with your Gmail username and password for sending alerts.

## **Usage**

### **Data Preprocessing:**
To preprocess the review data (cleaning, tokenization, augmentation), run:

```bash
python data_preprocessing.py
```

### **Model Training and Performance Monitoring:**

Start training the model and monitor performance via **Wandb.ai** and **MLflow**:

```bash
python train_model.py
```

- The script will log metrics to both **Wandb.ai** and **MLflow**.
- **MLflow** will automatically detect and check for model staleness, comparing it to past performance.
- **Checkpoints** will continuously monitor performance metrics, and if performance degrades, the model will be retrained, and an alert will be sent.

### **Inference Requests:**

The project supports two types of inference:

#### **1. Batch Inference:**
To process large batches of reviews at once:

```bash
python batch_inference.py --input_path path_to_input_file.csv --output_path path_to_output_file.csv
```

#### **2. Live Inference via Flask:**
Start the Flask server for live inference requests:

```bash
python app.py
```

Sample JSON payload:

```json
{
  "review": "The movie was fantastic and I loved it!"
}
```

The server will return a prediction:

```json
{
  "sentiment": "positive"
}
```

### **wandb Dashboard:**

To monitor the model's metrics and check for staleness, launch the **wandb.ai UI**:

 

Navigate to `https://wandb.ai/aashoksai306-jeppiaar-engineering-college/huggingface/runs/ujprk3ab/overview` in your browser to visualize your experiment metrics.

### **Alerts and Automatic Retraining:**

- If the model staleness check detects outdated performance, an email alert is triggered via **Gmail**.
- **MLflow** will automatically start retraining the model using new hyperparameters or data.

---

## **Model Performance**

Below are the graphs depicting the model's training and evaluation performance:

![Training vs Validation Accuracy](./images/training_validation_accuracy.png)

![Wandb and MLflow Performance Metrics](./images/performance_metrics.png)

### **MLflow Model Staleness Check:**
When the **wandbai** dashboard detects staleness, a notification is sent, and the model is retrained. Below is an example of a stale model retraining cycle.

![MLflow Model Staleness](./images/mlflow_staleness_check.png)

---

## **Deployment with Flask**

The model is deployed using **Flask**, allowing live predictions through HTTP requests. The API supports two endpoints:
- `/predict`: For real-time sentiment prediction on individual reviews.
- `/batch_predict`: For batch processing of reviews in bulk.

---

## **Alert System**

The system sends automated email alerts when the model's performance declines. Below is a sample of the email alert:

![Email Alert](./images/email_alert.png)

---

 
## **Conclusion**

This project provides a highly scalable and automated **Sentiment Analysis** solution. It continuously monitors model performance using **MLflow** and **Wandb.ai**, with automated retraining mechanisms. The deployment via **Flask** supports both real-time and batch inferences, making this a flexible tool for various sentiment analysis applications.

---

## **Contributors**

- **Ashok Sai G**

---

### **How to Contribute**

We welcome contributions! Fork the repository, create a feature branch, and submit a pull request to enhance the project’s functionality or improve performance.

---

### **References**

1. **Transformer Model for Sentiment Analysis:**
   - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is All You Need*. In Advances in Neural Information Processing Systems (pp. 5998-6008).

2. **MLflow for Model Tracking and Staleness Detection:**
   - MLflow Documentation: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

3. **Wandb.ai for Experiment Tracking:**
   - Biewald, L. (2020). Experiment Tracking with Weights and Biases. Available at [wandb.ai](https://wandb.ai/)

4. **Flask for Local Model Deployment:**
   - Flask Documentation: [https://flask.palletsprojects.com/en/2.1.x/](https://flask.palletsprojects.com/en/2.1.x/)

5. **Email Alerts Using SMTP:**
   - Gmail SMTP Documentation: [https://support.google.com/mail/answer/7126229](https://support.google.com/mail/answer/7126229)

---
 
