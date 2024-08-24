from torch.utils.data.dataloader import DataLoader
from .. import MODEL_DIR
import torch.nn as nn
import torch
import os

class TRAIN:

    @staticmethod
    def run(gptmodel: nn.Module, dataloader: DataLoader, batch_size: int,
            filename: str = 'model_weights.pth',
            epochs: int = 100, lr=0.001, momentum=0.9):

        model_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(model_path):
            gptmodel.load_state_dict(torch.load(model_path))

        optimizer = torch.optim.SGD(gptmodel.parameters(), lr=lr, momentum=momentum)
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

        best_loss = float('inf')

        for e in range(epochs):
            running_loss = 0

            ds = next(iter(dataloader))

            input_ids = torch.stack(ds['input_ids']).T
            label = torch.stack(ds['label']).T
            # We don't need this as it's already masked in selfattn.py line 23 but needed for loss.
            attn = torch.stack(ds['attention_mask']).T

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = gptmodel(input_ids)

            # Compute the loss and its gradients
            first = True

            for i in range(batch_size):
                if first:
                    first = False
                    loss = loss_fn(outputs[i][attn[i] == 1], label[i][attn[i] == 1])
                else:
                    loss += loss_fn(outputs[i][attn[i] == 1], label[i][attn[i] == 1])

            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if best_loss > running_loss:
                torch.save(gptmodel.state_dict(),
                           model_path)

                best_loss = running_loss

            print(best_loss)



