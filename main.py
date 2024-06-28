import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

os.environ["KAGGLE_KEY"] = "ciyiwakana"
os.environ["KAGGLE_USERNAME"] = "eed3361f08b8a842b558dfeb7cfd6213"

!kaggle datasets download -d mahmoudreda55/satellite-image-classification

!unzip "satellite-image-classification.zip"

base_dir = 'data'

def compute(image_path, bins=32, size=(64, 64), return_separate_channels=True):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size)
    channels_hist = []
    for channel in range(3):  # For RGB channels
        channel_hist = np.histogram(image.getchannel(channel), bins=bins, range=[0, 256])[0]
        channel_hist = channel_hist / np.sum(channel_hist)  # Normalize histogram
        channels_hist.append(channel_hist)
    return np.array(channels_hist)

def load_dataset(base_dir):
    categories = ['cloudy', 'desert', 'green_area', 'water']
    features = []
    labels = []
    mean_rgbs = []  # Store mean RGB values for plotting
    for label, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img_features = compute(img_path)
            image = Image.open(img_path).convert('RGB')
            mean_rgb = np.mean(np.array(image.resize((64, 64))), axis=(0, 1))
            features.append(img_features.flatten())  # Flatten to 1D
            labels.append(label)
            mean_rgbs.append(mean_rgb)
    return np.array(features), np.array(labels), np.array(mean_rgbs)

def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

def knn_predict(training_features, training_labels, test_instance, k=5):
    distances = []
    for i in range(len(training_features)):
        dist = euclidean_distance(test_instance, training_features[i])
        distances.append((training_labels[i], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = distances[:k]

    votes = {}
    for neighbor in neighbors:
        response = neighbor[0]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    return sorted_votes[0][0]

def split_dataset(features, labels, mean_rgbs, test_size=0.2):
    total_samples = len(features)
    test_samples = int(total_samples * test_size)
    indices = np.random.permutation(total_samples)
    training_idx, test_idx = indices[test_samples:], indices[:test_samples]
    training_features, test_features = features[training_idx,:], features[test_idx,:]
    training_labels, test_labels = labels[training_idx], labels[test_idx]
    training_mean_rgbs, test_mean_rgbs = mean_rgbs[training_idx], mean_rgbs[test_idx]
    print(f"Number of Training samples {training_features.shape[0]}\n, Number of Testing samples {test_features.shape[0]}")
    return training_features, training_labels, test_features, test_labels, training_mean_rgbs, test_mean_rgbs

def plot_rgb_3d(mean_rgbs, labels, title, subplot_position):
    ax = fig.add_subplot(subplot_position, projection='3d')
    categories = ['cloudy', 'desert', 'green_area', 'water']
    colors = ['skyblue', 'orange', 'green', 'blue']
    for i, category in enumerate(categories):
        idx = labels == i
        ax.scatter(mean_rgbs[idx, 0], mean_rgbs[idx, 1], mean_rgbs[idx, 2], label=category, c=colors[i])
    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Blue Channel')
    ax.set_title(title)
    ax.legend()


def evaluate_knn(base_dir, k=5):
    features, labels, mean_rgbs = load_dataset(base_dir)
    training_features, training_labels, test_features, test_labels, training_mean_rgbs, test_mean_rgbs = split_dataset(features, labels, mean_rgbs)
    global fig
    fig = plt.figure(figsize=(20, 7))
    plot_rgb_3d(training_mean_rgbs, training_labels, "Training Samples RGB Distribution", 131)
    plot_rgb_3d(test_mean_rgbs, test_labels, "Testing Samples RGB Distribution", 132)
    predictions = []
    for test_instance in test_features:
        prediction = knn_predict(training_features, training_labels, test_instance, k)
        predictions.append(prediction)
    plot_rgb_3d(test_mean_rgbs, np.array(predictions), "Predicted Classes for Testing Samples", 133)
    plt.show()
    conf_matrix = confusion_matrix(test_labels, predictions)

    # Visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix, cmap='coolwarm')  # You can change the colormap to others such as 'Blues', 'viridis', etc.
    plt.title('Confusion Matrix')
    fig.colorbar(cax)

    # Add annotations for cell values
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if conf_matrix.max() > 0 and val > (conf_matrix.max() / 2.) else 'black')

    ticks = np.arange(len(np.unique(test_labels)))
    plt.xticks(ticks, np.unique(test_labels))
    plt.yticks(ticks, np.unique(test_labels))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions))
    precision, recall, fscore, _ = score(test_labels, predictions, average='weighted')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", fscore)
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("Classification Report:\n", classification_report(test_labels, predictions))


def predict_image_class(image_path, base_dir, k=3):
    training_features, training_labels, _ = load_dataset(base_dir)
    test_features = compute(image_path).flatten()  # Flatten the features to match the training data
    predicted_class = knn_predict(training_features, training_labels, test_features, k)
    categories = ['cloudy', 'desert', 'green_area', 'water']
    return categories[predicted_class]


evaluate_knn(base_dir, k=67)
