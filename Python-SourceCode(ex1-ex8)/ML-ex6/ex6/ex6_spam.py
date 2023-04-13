import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import processEmail as pe
import emailFeatures as ef

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

print('Preprocessing sample email (emailSample1.txt) ...')

file_contents = open('emailSample1.txt', 'r').read()
word_indices = pe.process_email(file_contents)

# Print stats
print('Word Indices: ')
print(word_indices)

input('Program paused. Press ENTER to continue')

print('Extracting Features from sample email (emailSample1.txt) ... ')

# Extract features
features = ef.email_features(word_indices)

# Print stats
print('Length of feature vector: {}'.format(features.size))
print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))

input('Program paused. Press ENTER to continue')

data = scio.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].flatten()

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)')

c = 0.1
clf = svm.SVC(c, kernel='linear')
clf.fit(X, y)

p = clf.predict(X)

print('Training Accuracy: {}'.format(np.mean(p == y) * 100))

# Load the test dataset
data = scio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('Evaluating the trained linear SVM on a test set ...')

p = clf.predict(Xtest)

print('Test Accuracy: {}'.format(np.mean(p == ytest) * 100))

input('Program paused. Press ENTER to continue')

vocab_list = pe.get_vocab_list()
indices = np.argsort(clf.coef_).flatten()[::-1]
print(indices)

for i in range(15):
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf.coef_.flatten()[indices[i]]))

input('ex6_spam Finished. Press ENTER to exit')
