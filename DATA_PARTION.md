# 1. How the Dataset is Divided (data/mnist_loader.py)
Before any training begins, the central server downloads the standard MNIST training dataset (60,000 images) and the testing dataset (10,000 images).

The training dataset is then divided among the 5 clients using an IID (Independent and Identically Distributed) split:

Shuffling: The indices of all 60,000 training images are completely randomized (shuffled) to ensure there is no ordering bias.
Partitioning: The shuffled indices are split evenly into 5 equal-sized shards.
Distribution: Each of the 5 clients is assigned exactly one shard. Since there are 60,000 total training images and 5 clients, every client gets exactly 12,000 unique training images.
Because the split is IID (randomized), each client's 12,000 images contain a roughly equal mix of all 10 digits (0-9).

# 2. How Client Training Happens (clients/benign_client.py)
For each of the 3 rounds in Phase 1, the following process occurs:

Model Distribution: The central server sends the current global weights to all 5 clients.
Local Initialization: Each client creates a local copy of the neural network and loads the global weights into it. They also initialize a Stochastic Gradient Descent (SGD) optimizer.
Local Epochs: Every client trains on its own 12,000 images for a set number of local_epochs (configured as 2 epochs by default).
The client feeds batches of 64 images through the network.
It calculates the Cross-Entropy Loss between the predictions and the actual labels.
It computes the gradients and updates its local model weights.
During this process, it keeps a running tally of its training accuracy and loss.
Sending Updates: After finishing its local epochs, the client does not send its raw data to the server (preserving privacy). Instead, it packages its newly updated model weights, along with its training metadata (accuracy, loss, sample count), into a ModelUpdate object and sends that back to the central server.
Aggregation: Once the server receives all 5 ModelUpdate objects, it uses Federated Averaging (FedAvg) to average the weights together, producing a new global model for the next round.
Because all 5 clients are acting honestly in Phase 1, the Anomaly Detector is effectively bypassed—no clients are flagged, and all 5 updates are included in the average.