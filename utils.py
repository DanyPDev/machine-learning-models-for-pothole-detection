"""Library implementing utility functions for pothole_detection.ipynb.

Authors
 * Dany Pham 2023
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def get_dataset(convert_to_grayscale=True, downscale = True):
    """Returns train, validation and test set (70%, 10%, 20% of total dataset respectively) as a numpy array.
        6 numpy array are returned with shapes (2817, 300, 400), (2817,), (403, 300, 400),
        (403,), (806, 300, 400), (806,) respectively.

    Arguments
    ---------
    convert_to_grayscale : bool
        Indicates wheter or not to convert image to greyscale.
    downscale : bool
        Indicates wheter or not to downscale the pothole images.

    Example
    -------
    >>> X_train, y_train, X_test, y_test, X_validation, y_validation = get_dataset(True)
    >>> X_train.shape
    (2817, 300, 400)
    """
    # Contains all image ids and their label
    dataset_imageid_to_label_data = np.loadtxt(
        "train_ids_labels.csv",    
        dtype=np.str_ ,
        delimiter=",",
        skiprows=1)

    X, y = [], []

    for id, label in dataset_imageid_to_label_data:
      file_name = "all_data//all_data//" + id + ".JPG"
      image_matrix = cv2.cvtColor(img.imread(file_name), cv2.COLOR_RGB2GRAY) if convert_to_grayscale else img.imread(file_name)
      if downscale:
          image_matrix = cv2.resize(image_matrix, (0,0), fx=0.5, fy=0.5)
      X.append(image_matrix)
      y.append(int(label))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)
    
    X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(
        X_train, y_train, train_size=0.875, test_size=0.125, random_state=0) 

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_validation), np.array(y_validation)


def plot_matrix_image_greyscale(X, y, num_display=4):
    """This function displays greyscaled images on the screen (must be greyscaled).
       Nothing is returned.

    Arguments
    ---------
    X : numpy array
        Input features of shape (number_of_images, width, length)
        Where width = 300 and length = 400.
    y : numpy array
        Labels for X of shape (number_of_images,).
    num_display : int
        Indicates how many images to display.

    Example
    -------
    >>> X_train, y_train, X_test, y_test, X_validation, y_validation = get_dataset(True)
    >>> X_train.shape
    (2817, 300, 400)
    >>> plot_matrix_image_greyscale(X_train, y_train, num_display=4)  # display 4 images on the screen
    """
    k, m, n = X.shape
    V, labels = X[:num_display], y[:num_display]     
    vmin, vmax = np.percentile(V, [0.1, 99.9])                    
    for ith in range(num_display):
        plt.figure(figsize = (m/30,n/30))
        plt.colorbar(plt.imshow(V[ith], vmin=vmin, vmax=vmax, cmap=plt.get_cmap('gray')), shrink=n/900)
        if labels[ith] == 0:
            plt.title("Does Not Contain Potholes", fontsize=20)
        else:
            plt.title("Contains Potholes", fontsize=20)


def display_confusion_matrix(y_test, predictions, title):
    """This function displays a confusion matrix given label and predicted labels
       Taken from scikit-learn documentation and stack overflow
       Source 1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
       Source 2: https://stackoverflow.com/questions/66483409/adjust-size-of-confusionmatrixdisplay-scikitlearn
       No return value.

    Arguments
    ---------
    y_test : numpy array
        Label array of shape (number_of_images,).
    predictions : numpy array
        Label array of shape (number_of_images,).
    title : str
        Title of the confusion matrix to be displayed

    Example
    -------
    >>> X_train, y_train, X_test, y_test, X_validation, y_validation = get_dataset(True)
    >>> X_train.shape
    (2817, 300, 400)
    >>> y_train.shape
    (2817,)
    >>> mlp = MLP()
    >>> y_pred = MLP(X_train.reshape(-1,120000))  # predict using a model
    >>> y_pred.shape
    (2817,)
    >>> display_confusion_matrix(y_train, y_train_pred, "MLP confusion matrix with training data")
    # confusion matrix is displayed on screen
    """
  
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax = ax)
    plt.title(title)
    plt.show()

def bce_loss(y, y_pred):
    """Returns binary cross entropy loss, a float is returned

    Arguments
    ---------
    y : numpy array
        Label array of shape (number_of_images,).
    y_pred : numpy array
        Prediction labels array of shape (number_of_images,).

    Example
    -------
    >>> X_train, y_train, X_test, y_test, X_validation, y_validation = get_dataset(True)
    >>> X_train.shape
    (2817, 300, 400)
    >>> y_train.shape
    (2817,)
    >>> mlp = MLP()
    >>> y_pred = MLP(X_train.reshape(-1,120000))  # predict using a model
    >>> y_pred.shape
    (2817,)
    >>> bce_loss(y_train, y_pred)  # BCE loss of train prediction
    1.5140 ...
    """
    return log_loss(y, y_pred, normalize=True)