from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()

images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5), facecolor='pink')
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
plt.show()

import random
from sklearn import ensemble

n_samples = len(digits.images)
x = digits.images.reshape((n_samples,-1))
y = digits.target

sample_index = random.sample(range(len(x)),len(x)//5)
valid_index=[i for i in range(len(x)) if i not in sample_index]

sample_images = [x[i] for i in sample_index]
valid_images = [x[i] for i in valid_index]

sample_target = [y[i] for i in sample_index]
valid_target = [y[i] for i in valid_index]

classifier = ensemble.RandomForestClassifier()
classifier.fit(sample_images, sample_target)

score = classifier.score(valid_images, valid_target)
print ('Random Tree Classifier')
print ('score: ',str(score))