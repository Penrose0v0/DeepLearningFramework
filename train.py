import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import argparse
import time
import traceback

from network import Net
from loss import NetLoss
from dataset import NetDataset
from utils import Logger, setup_seed, draw_figure, convert_seconds

# 3407 is all you need
setup_seed(3407)
fmt = "----- {:^25} -----"


class Trainer:
    def __init__(self, args):
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dataset_path = args.dataset_path
        self.model_path = args.model_path
        self.save_path = args.save_path
        self.log_path = args.log_path

        # Set device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Create neural network
        self.model = self.create_neural_network()

        # Define criterion and optimizer
        self.criterion = NetLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Load dataset
        self.train_loader = self.load_dataset()

    def run(self):
        print(fmt.format("Start training") + '\n')
        min_loss = -1
        best_epoch = 0
        epoch_list, loss_list = [], []
        very_start = time.time()
        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(very_start))
        self.log = Logger(os.path.join(self.log_path, f"{formatted_time}.txt"))

        try:
            for epoch in range(self.epochs):
                start = time.time()

                # Train one epoch
                current_loss = self.train(epoch)

                # Save model
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "current.pth"))
                if current_loss < min_loss or min_loss == -1:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "best.pth"))
                    self.log("Update the best model")
                    min_loss = current_loss
                    best_epoch = epoch + 1

                # Draw figure
                epoch_list.append(epoch + 1)
                loss_list.append(current_loss)
                draw_figure(epoch_list, loss_list, "Loss", os.path.join(self.log_path, f"{formatted_time}.png"))

                # Elapsed time
                end = time.time()
                use_time = int(end - start)
                self.log(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")

        except Exception as e:
            self.log(traceback.format_exc())

        except KeyboardInterrupt:
            print()

        very_end = time.time()
        total_time = int(very_end - very_start)

        self.log(f"Training finished! Total elapsed time: {convert_seconds(total_time)}, "
                 f"Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")

    def train(self, epoch_num, count=100):
        self.model.train()
        running_loss, total_loss, total = 0.0, 0.0, 1
        self.log(f"< Epoch {epoch_num + 1} >")
        for batch_idx, data in enumerate(self.train_loader, 0):
            # Get data
            inputs = data
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            # Forward + Backward + Update
            outputs = self.model(inputs)
            loss = self.criterion(outputs)
            loss.backward()
            self.optimizer.step()

            # Calculate loss
            running_loss += loss.item()
            total_loss += loss.item()
            if batch_idx % count == count - 1:
                print('\r', end='')
                self.log('Batch %d\t loss: %.6f' % (batch_idx + 1, running_loss / count))
                running_loss = 0.0
            else:
                print(f"\r[{batch_idx % count + 1} / {count}]", end='')
            total += 1
        print('\r', end='')

        return total_loss / total

    def create_neural_network(self):
        print(fmt.format("Create neural network"))
        model = Net().to(self.device)
        if self.model_path != '':
            print(f"Loading pretrained model: {self.model_path}")
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(self.model_path).items()})
        else:
            print("Creating new model")

        device_count = torch.cuda.device_count()
        print(f"Using {device_count} GPUs")
        model = nn.DataParallel(model, device_ids=[i for i in range(device_count)])
        print()

        return model

    def load_dataset(self):
        print(fmt.format("Load dataset"))
        train_set = NetDataset(folder=self.dataset_path)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        print()
        return train_loader


if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')

    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)

    parser.add_argument('--dataset-path', type=str, default='dataset')
    parser.add_argument('--save-path', type=str, default='./weights/')
    parser.add_argument('--log-path', type=str, default='./logs/')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
