import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sens as sens
from IPython.core.display import display
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, \
    precision_recall_curve, average_precision_score
import ipywidgets as widgets
from IPython.display import clear_output

df = pd.read_csv('son.csv')

print("Total number of rows : " + str(len(df)))

print("We see the first 10 cases in this table.")

print(df.head(10))

threshold = 0.6

true_positives = ((df.y_pred > threshold) & (df.y_test == 1)).sum()
false_positives = ((df.y_pred > threshold) & (df.y_test == 0)).sum()
true_negatives = ((df.y_pred <= threshold) & (df.y_test == 0)).sum()
false_negatives = ((df.y_pred <= threshold) & (df.y_test == 1)).sum()

print("True positives: " + str(true_positives))
print("False positives: " + str(false_positives))
print("True negatives: " + str(true_negatives))
print("False negatives: " + str(false_negatives))

cm = np.array([[true_positives, false_negatives],
               [false_positives, true_negatives]])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    cm = np.array([[true_positives, false_negatives],
                   [false_positives, true_negatives]])

    # This is the definition of a function to plot the confusion matrix
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
plot_confusion_matrix(cm, classes=['Autistic Spectrum Disorder', 'Normal'],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=['Autistic Spectrum Disorder', 'Normal'], normalize=True,
                          title='Normalized confusion matrix')

plt.show()

sensitivity = true_positives / (true_positives + false_negatives)
print("Sensitivity: " + str(sensitivity))

specificity = true_negatives / (true_negatives + false_positives)
print("Specificity: " + str(specificity))

fnr = false_negatives / (true_positives + false_negatives)
print("False Negative Rate: " + str(fnr))

ppv = true_positives / (true_positives + false_positives)
print("Positive Predictive Value: " + str(ppv))

npv = true_negatives / (true_negatives + false_negatives)
print("Negative Predictive Value: " + str(npv))

accuracy = (true_positives + true_negatives) / (true_positives + false_negatives + true_negatives + false_positives)
print("Accuracy: " + str(accuracy))

f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
print("F1-score: " + str(f1))

slider = widgets.IntSlider(value=50, min=0.0, max=100, step=1 )
display(slider)

button = widgets.Button(description="Calculate")
output = widgets.Output()

slider = widgets.IntSlider(value=50, min=0.0, max=100, step=1)
display(slider)

button = widgets.Button(description="Calculate")
output = widgets.Output()


def on_button_clicked(b):
    # Display the message within the output widget.
    with output:
        clear_output()
        threshold = slider.value / 100.
        print("Threshold: " + str(threshold))
        true_positives = ((df.y_pred > threshold) & (df.y_test == 1)).sum()
        false_positives = ((df.y_pred > threshold) & (df.y_test == 0)).sum()
        true_negatives = ((df.y_pred <= threshold) & (df.y_test == 0)).sum()
        false_negatives = ((df.y_pred <= threshold) & (df.y_test == 1)).sum()
        cm = np.array([[true_positives, false_negatives],
                       [false_positives, true_negatives]])

        plot_confusion_matrix(cm, classes=['Autistic Spectrum Disorder', 'Normal'], normalize=True,
                              title='Normalized confusion matrix')
        plt.show()

        sensitivity = true_positives / (true_positives + false_negatives)
        print("Sensitivity: " + str(sensitivity))
        specificity = true_negatives / (true_negatives + false_positives)
        print("Specificity: " + str(specificity))
        precision = true_positives / (true_positives + false_positives)
        print("Precision: " + str(precision))
        accuracy = (true_positives + true_negatives) / (
                    true_positives + false_negatives + true_negatives + false_positives)
        print("Accuracy: " + str(accuracy))
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        print("F1-score: " + str(f1))




tp, tn, fp, fn = [], [], [], []
sens, spec, acc, f1, prec = [], [], [], [], []
th = []

for threshold in range(100):
    threshold/=100
    tp.append( ((df.y_pred > threshold) & (df.y_test == 1)).sum())
    fp.append( ((df.y_pred > threshold) & (df.y_test == 0)).sum())
    tn.append( ((df.y_pred <= threshold) & (df.y_test == 0)).sum())
    fn.append( ((df.y_pred <= threshold) & (df.y_test == 1)).sum())
    sens.append(tp[-1] / (tp[-1] + fn[-1]))
    spec.append(tn[-1] / (tn[-1] + fp[-1]))
    acc.append((tp[-1] + tn[-1]) / (tp[-1] + tn[-1] + fp[-1] + fn[-1]))
    f1.append((2 * tp[-1]) / (2 * tp[-1] + fn[-1] + fp[-1]))
    if (tp[-1] + fp[-1]) == 0:
        prec.append(1e+6)  # avoid divide by 0 error
    else:
        prec.append(tp[-1] / (tp[-1] + fp[-1]))
    th.append(threshold)

sens = np.array(sens)
spec = np.array(spec)
prec = np.array(prec)
th = np.array(th)

plt.figure(figsize=(10,7))
plt.plot(th, sens, ls='-.')
plt.plot(th, spec, ls='--')
plt.plot(th, acc, ls='-')
plt.plot(th, f1, ls=':')
plt.legend(['Sensitivity', 'Specificity', 'Accuracy', 'F1-score'], fontsize=12)
plt.xlabel('Threshold', size=15)
plt.ylabel('Metric', size=15)
plt.show()


slider = widgets.IntSlider(value=50, min=0.0, max=100, step=1 )
display(slider)
threshold = slider.value / 100.
ind = abs(th - threshold).argmin()
roc_auc = average_precision_score( df.y_test, df.y_pred)

print('Auc Roc value:{} '.format(roc_auc))
