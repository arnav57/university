from classes import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 1500
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

nn = neuralnet(784, 10, 'classifier')

iter_count = 0
max_iter = 1000

accs = []
for img_batch, label_batch in train_loader:
    succ = 0
    # obtain single image and label
    for i in range(img_batch.shape[0]):
        img = img_batch[i].reshape(-1, 1) # needed to flatten this for use in a MLP
        label = one_hot_encode(label_batch[i].numpy()) # this is now one hot encoded

        res = nn.forward(img)

        pred_label = np.argmax(res)
        actual_label = np.argmax(label)
        if (pred_label == actual_label):
            succ += 1

        nn.backward(img, label, 2e-2)

    print(f'Batch {iter_count} -  Train Accuracy: {succ/batch_size}')
    accs.append(succ/batch_size)

    # break out of loop
    iter_count += 1
    if (iter_count  > max_iter):
        break

it = range(len(accs))
print(len(it))
print(len(accs))

plt.plot(it, accs)
plt.title('Train Acc vs Iteration')
plt.ylabel('Train Accuracy')
plt.xlabel('Iteration')
plt.show()

print('Testing')

accs = []
for img_batch, label_batch in test_loader:
    succ = 0
    # obtain single image and label
    for i in range(img_batch.shape[0]):
        img = img_batch[i].reshape(-1, 1) # needed to flatten this for use in a MLP
        label = one_hot_encode(label_batch[i].numpy()) # this is now one hot encoded

        res = nn.forward(img)

        pred_label = np.argmax(res)
        actual_label = np.argmax(label)
        if (pred_label == actual_label):
            succ += 1

        # nn.backward(img, label, 1e-2)

    print(f'Batch {iter_count} -  Test Accuracy: {succ/batch_size}')
    accs.append(succ/batch_size)

    # break out of loop
    iter_count += 1
    if (iter_count  > max_iter):
        break

print(f'Total Average Test Accuracy: {np.mean(accs)}')


# nn = neuralnet()
# label = np.array([0,0,0,0,0,0,0,1,0,0]).reshape(10,1)
# input = np.random.randn(784, 1)
# res = nn.forward(input)
# nn.backward(input, label, 1e-1)
# print(res)