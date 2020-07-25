"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import json
import time
import numpy as np
import requests
import torch.optim as optim
from util import set_experiment_id, write_out, write_model_to_disk, write_result_summary


def train_model(data_set_identifier, model, train_loader, validation_loader,
                learning_rate, minibatch_size=64, eval_interval=50, hide_ui=False,
                use_gpu=False, minimum_updates=1000,
                optimizer_type='adam', restart=False):
    set_experiment_id(data_set_identifier, learning_rate, minibatch_size)

    validation_dataset_size = validation_loader.dataset.__len__()

    if use_gpu:
        model = model.cuda()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if restart:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=32)

    sample_num = list()
    train_loss_values = list()
    train_drmsd_values = list()
    validation_loss_values = list()
    validation_angles_loss_values = list()
    best_model_loss = 1e20
    best_model_minibatch_time = None
    best_model_path = None
    best_json_data = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
    # for i in range(2):
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)
        drmsd_tracker = np.zeros(0)
        for _minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            start_compute_loss = time.time()
            loss, drmsd_avg = model.compute_loss(training_minibatch)
            write_out("Train loss:", float(loss))
            start_compute_grad = time.time()
            drmsd_avg.backward()
            loss_tracker = np.append(loss_tracker, float(loss))
            drmsd_tracker = np.append(drmsd_tracker, float(drmsd_avg))
            end = time.time()
            write_out("Loss time:", start_compute_grad - start_compute_loss, "Grad time:",
                      end - start_compute_grad)
            optimizer.step()
            if restart:
                scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

            # for every eval_interval samples, plot performance on the validation set
            if minibatches_proccesed % eval_interval == 0:

                write_out("Testing model on validation set...")

                train_loss = float(loss_tracker.mean())
                train_drmsd = float(drmsd_tracker.mean())
                loss_tracker = np.zeros(0)
                drmsd_tracker = np.zeros(0)
                validation_loss, json_data, _, validation_angles_loss = model.evaluate_model(validation_loader)

                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)
                    best_json_data = json_data

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss, "Train drmsd:", train_drmsd)
                write_out("Best model so far (validation loss): ", best_model_loss, "at time",
                          best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Minibatches processed:", minibatches_proccesed)
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(train_loss)
                train_drmsd_values.append(train_drmsd)
                validation_loss_values.append(validation_loss)
                validation_angles_loss_values.append(validation_angles_loss)
                json_data["validation_dataset_size"] = validation_dataset_size
                json_data["sample_num"] = sample_num
                json_data["train_loss_values"] = train_loss_values
                json_data["train_drmsd_values"] = train_drmsd_values
                json_data["validation_loss_values"] = validation_loss_values
                json_data['validation_angles_loss_values'] = validation_angles_loss_values

                write_out(json_data)

                if not hide_ui:
                    res = requests.post('http://localhost:5000/graph', json=json_data)
                    if res.ok:
                        print(res.json())

                if minibatches_proccesed > minimum_updates and minibatches_proccesed \
                        >= best_model_minibatch_time + minimum_updates:
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    write_result_summary(json.dumps(best_json_data))
    return best_model_path
