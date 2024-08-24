import torch.nn as nn

loss = nn.CrossEntropyLoss(reduction='mean')
running_sum = 0

for i in range(30):
    running_sum += loss(test[i], label[i])
    print(loss(test[i], label[i]))