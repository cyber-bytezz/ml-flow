import matplotlib.pyplot as plt

def plot_metrics():
    # Placeholder for plotting metrics
    epochs = [1, 2, 3]
    accuracy = [0.85, 0.90, 0.95]
    loss = [0.5, 0.3, 0.1]

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Accuracy')
    ax.plot(epochs, loss, label='Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.legend()
    plt.title('Model Performance Over Epochs')
    plt.show()
