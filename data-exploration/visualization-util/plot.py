import matplotlib.pyplot as plt

def visualize_feature_frequency(data, featureName):
    """
    Graph the frequency of a feature in the dataset
    """
    featureFrequency = data[featureName].value_counts()
    featureFrequency.plot(kind='bar')
    plt.title('Frequency of ' + featureName)
    plt.xlabel(featureName)
    plt.ylabel('Frequency')
    plt.show()