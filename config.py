# Network parameters
learning_rate = 0.001

num_input = 28 ** 2 # 28x28 bitmaps
num_1 = 100
num_2 = 50
num_output = 10

# Run parameters
use_gpu = True

# Training parameters
# (affects next run of train.py)
batch_size = 10000
num_epochs = 100

# File params
# affects train, guess_bmp
checkpoint_fname = "model-adam-morenodes-scaled.ckpt"
guess_fname = "test.bmp"