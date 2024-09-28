import matplotlib.pyplot as plt
import json

def plot_metrics():
    # Load saved metrics
    with open('logs/metrics.json', 'r') as f:
        metrics = json.load(f)

    epochs = metrics['epochs']
    accuracy = metrics['accuracy']
    loss = metrics['loss']

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Accuracy')
    ax.plot(epochs, loss, label='Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.legend()
    plt.title('Model Performance Over Epochs')
    plt.show()
