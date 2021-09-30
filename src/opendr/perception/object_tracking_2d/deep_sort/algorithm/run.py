# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# train function is based on deep_sort/deep/train.py

import os
import torch
import time
from opendr.perception.object_tracking_2d.deep_sort.algorithm.deep_sort.deep.model import Net
from opendr.engine.datasets import MappedDatasetIterator
from opendr.perception.object_tracking_2d.logger import Logger


def train(
    net: Net, dataset, epochs, iters, val_dataset, optimizer,
    batch_size, val_epochs, train_transforms, val_transforms,
    device, log, log_interval, checkpoints_path, checkpoint_after_iter
):

    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        MappedDatasetIterator(dataset, lambda data: (train_transforms(data[0]), data[1])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        MappedDatasetIterator(val_dataset, lambda data: (val_transforms(data[0]), data[1])),
        batch_size=batch_size,
        shuffle=True,
    )

    training_loss = 0.0
    train_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    last_val_loss = -1

    if iters == -1:
        iters = len(train_loader)

    iter = 0

    for epoch in range(epochs):

        net.train()

        for idx, (inputs, labels) in enumerate(train_loader):

            if idx > iters:
                continue

            # forward
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumurating
            training_loss += loss.item()
            train_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

            # print
            if (idx + 1) % log_interval == 0:
                end = time.time()
                log(
                    Logger.LOG_WHEN_NORMAL,
                    "[{}/{}][progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                        epoch + 1,
                        epochs,
                        100.0 * (idx + 1) / len(train_loader),
                        end - start,
                        training_loss / log_interval,
                        correct,
                        total,
                        100.0 * correct / total,
                    )
                )
                training_loss = 0.0
                start = time.time()

            if checkpoint_after_iter > 0:
                if ((iter + 1) % checkpoint_after_iter == 0):
                    torch.save(net.state_dict(), os.path.join(
                        checkpoints_path,
                        f"checkpoint_{iter}.pth"
                    ))

        if val_epochs > 0 and (epoch + 1) % val_epochs == 0:
            log(Logger.LOG_WHEN_NORMAL, "Eval")
            net.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            start = time.time()
            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(val_loader):

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    correct += outputs.max(dim=1)[1].eq(labels).sum().item()
                    total += labels.size(0)

                end = time.time()
                log(
                    Logger.LOG_WHEN_NORMAL,
                    "[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                        100.0 * (idx + 1) / len(val_loader),
                        end - start,
                        test_loss / len(val_loader),
                        correct,
                        total,
                        100.0 * correct / total,
                    )
                )

            last_val_loss = test_loss

    result = {"last_val_loss": last_val_loss}

    return result
