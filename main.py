import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
import traceback

from network import Net
from loss import NetLoss
from dataset import NetDataset
from utils import Logger, setup_seed, draw_figure, convert_seconds

# 3407 is all you need
setup_seed(3407)

def train(epoch_num, count=100):
    model.train()
    running_loss, total_loss, total = 0.0, 0.0, 1
    log(f"< Epoch {epoch_num + 1} >")
    for batch_idx, data in enumerate(train_loader, 0):
        # Get data
        inputs = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # Forward + Backward + Update
        outputs = model(inputs)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()

        # Calculate loss
        running_loss += loss.item()
        total_loss += loss.item()
        if batch_idx % count == count - 1:
            log('Batch %d\t loss: %.6f' % (batch_idx + 1, running_loss / count))
            running_loss = 0.0
        total += 1

    return total_loss / total


if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--dataset-path', type=str, default='dataset')
    args = parser.parse_args()

    # Set hyper-parameters
    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dataset_path = args.dataset_path
    model_path = args.model_path

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create neural network
    print(fmt.format("Create neural network"))
    model = Net()
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPUs")

    # Load pretrained model or create a new model
    if model_path != '':
        print(f"Loading pretrained model: {model_path}")
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    else:
        print("Creating new model")
    model = nn.DataParallel(model)
    model.to(device)
    print()

    # Define criterion
    criterion = NetLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    # Load dataset
    print(fmt.format("Loading dataset"))
    train_set = NetDataset(folder=dataset_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    print()

    # Start training
    print(fmt.format("Start training") + '\n')
    min_loss = -1
    best_epoch = 0
    epoch_list, loss_list = [], []
    very_start = time.time()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(very_start))
    log = Logger(f"logs/{formatted_time}.txt")

    try:
        for epoch in range(epochs):
            start = time.time()

            # Train one epoch
            current_loss = train(epoch)

            # Save model
            torch.save(model.state_dict(), f"./weights/current.pth")
            if current_loss < min_loss or min_loss == -1:
                torch.save(model.state_dict(), f"./weights/best.pth")
                print("Update the best model")
                min_loss = current_loss
                best_epoch = epoch + 1

            # Draw figure and log
            epoch_list.append(epoch + 1)
            loss_list.append(current_loss)
            draw_figure(epoch_list, loss_list, "Loss", f"./logs/{formatted_time}.png")

            # Elapsed time
            end = time.time()
            use_time = int(end - start)
            log(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")

        very_end = time.time()
        total_time = int(very_end - very_start)

        log(f"Training finished! Total elapsed time: {convert_seconds(total_time)}, "
            f"Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")

    except Exception as e:
        log(traceback.format_exc())
