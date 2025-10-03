# 🍄 Super Mario Bros. PPO Agent 🍄

This project implements a Proximal Policy Optimization (PPO) agent to play Super Mario Bros. using PyTorch. It leverages parallel environments, custom reward shaping, and a convolutional neural network to train an agent capable of completing various levels.

## 🚀 Key Features

- **PPO Implementation:** Utilizes the Proximal Policy Optimization algorithm for reinforcement learning.
- **Parallel Environments:** Employs multiple parallel environments to accelerate training.
- **Custom Reward Shaping:** Implements a custom reward function to guide the agent's learning process, encouraging progress and penalizing failures.
- **Convolutional Neural Network:** Uses a CNN to process visual input from the game and extract relevant features.
- **Environment Wrappers:** Includes environment wrappers for preprocessing frames and customizing the reward function.
- **Evaluation Function:** Provides an evaluation function to assess the trained agent's performance.
- **TensorBoard Logging:** Logs training progress to TensorBoard for monitoring and analysis.
- **Model Saving:** Saves the trained model periodically for later use.

## 🛠️ Tech Stack

- **Frontend:** N/A (command-line interface)
- **Backend:** Python
- **Deep Learning Framework:** PyTorch
- **Reinforcement Learning Algorithm:** Proximal Policy Optimization (PPO)
- **Game Environment:** Gym Super Mario Bros
- **Environment Wrappers:** OpenAI Gym, Nes-Py
- **Image Processing:** OpenCV (cv2)
- **Numerical Computation:** NumPy
- **Parallel Processing:** `torch.multiprocessing`
- **Video Recording:** ffmpeg (optional)
- **Other:** `argparse`, `shutil`, `collections`

## 📦 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch (>=1.0)
- Gym Super Mario Bros
- Nes-Py
- OpenCV (cv2)
- NumPy
- TensorBoard (optional)

You can install the necessary dependencies using pip:

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install gym_super_mario_bros==7.3.0
pip install nes_py
pip install opencv-python
pip install numpy
pip install tensorboard
```

### Installation

1.  Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2.  (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

3.  Install the dependencies (if not already installed):

```bash
pip install -r requirements.txt # if you have a requirements.txt file
```

### Running Locally

1.  Navigate to the project directory.

2.  Run the `train.py` script with the desired arguments. For example:

```bash
python train.py --world 1 --stage 1 --saved_path ./trained_models --log_path ./tensorboard_logs
```

You can customize the training process by modifying the command-line arguments. Run `python train.py --help` to see a list of available options.

## 💻 Usage

The `train.py` script is the main entry point for training the PPO agent. It takes several command-line arguments that control the training process, such as the learning rate, discount factor, batch size, and number of training steps.

The `src/process.py` file contains the `eval` function, which can be used to evaluate the trained agent's performance.

The `src/env.py` file defines the environment wrappers and helper functions for interacting with the Super Mario Bros environment.

The `src/model.py` file defines the PPO neural network architecture.

## 📂 Project Structure

```
├── train.py          # Main training script
├── src
│   ├── model.py      # PPO model definition
│   ├── process.py    # Evaluation function
│   └── env.py        # Environment wrappers and creation
├── README.md         # This file
└── requirements.txt  # Project dependencies
```

## 📸 Screenshots

(Leave space for screenshots of the agent playing the game, TensorBoard logs, etc.)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.

## 📝 License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## 📬 Contact

If you have any questions or suggestions, please feel free to contact me at [your_email@example.com](mailto:your_email@example.com).

## 💖 Thanks

Thanks for checking out this project! I hope it's helpful for learning about reinforcement learning and PPO.


