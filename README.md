# PyDreamerV1: Clean pytorch implementation of Hafner et al Dreamer
<div align="center">
  <img src="./gif/boxing.gif" alt="Actual run in " width="200px" height="200px"/>
  <img src="./gif/quadruped.gif" alt="Actual run in " width="200px" height="200px"/>
  <img src="./gif/walker.gif" alt="Actual run in " width="200px" height="200px"/>
</div>
<div align="center">
  <img src="./gif/boxing_imagine.gif" alt="Imagination in " width="200px" height="200px"/>
  <img src="./gif/quadruped_imagine.gif" alt="Imagination in " width="200px" height="200px"/>
  <img src="./gif/walker_imagine.gif" alt="Imagination in " width="200px" height="200px"/>
</div>



This repository offers a comprehensive implementation of the Dreamer algorithm, as presented in the groundbreaking work by Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination." Our implementation is dedicated to faithfully reproducing the innovative approach of learning and planning within a learned latent space, enabling agents to efficiently master complex behaviors through imagination alone.

## Why Dreamer?

Dreamer stands at the forefront of model-based reinforcement learning by introducing an efficient method for learning behaviors directly from high-dimensional sensory inputs. It leverages a latent dynamics model to 'imagine' future states and rewards, enabling it to plan and execute actions that maximize long-term rewards purely from simulated experiences. This approach significantly improves sample efficiency over traditional model-free methods and opens new avenues for learning complex and nuanced behaviors in simulated environments. However, the official code was unfortunately regarded as complex and difficult to understand, and there are only a handful of Dreamer reimplementation that was able to reproduce the results.

## Implementation Highlights

- **Modular Design**: My implementation of the Recurrent State Space Model (RSSM) is broken down into cleanly separated modules for the transition, representation, and recurrent models. This not only facilitates a deeper understanding of the underlying mechanics but also allows for easy customization and extension.

- **True to the Source**: By closely adhering to the methodologies detailed in the original DreamerV1 paper, the code captures the essence of latent space learning and imagination-driven planning. From the incorporation of exploration noise to the td lambda calculation, every element is designed to replicate the paper's results as closely as possible. The sets of hyperparamenters are excactly indentical to the sets mentioned in the paper

- **Detailed Training Insights**: The training loop is separated and mirroring the paper's outline. Comprehensive comments of hidden implementation details thorough documentation accompany the code, serving as a valuable resource for both learning and further research.

## Getting Started

1. **Clone the Repository**: Get the code by cloning this repository to your local machine.
   ```
   git clone https://github.com/minhphd/PyDreamerV1
   ```

2. **Install Dependencies**: Ensure you have all necessary dependencies by running:
   ```
   pip3 install -r requirements.txt
   ```

3. **Run the Training**: Kickstart the training process with a simple command:
   ```
   python main.py --config <Path to config file>
   ```

4. **Visualize Results**: Utilize TensorBoard to observe training progress and visualize the agent's performance in real-time. Wandb is also supported, simply set enable to True and replace with your account information in config files.
   ```
   tensorboard --logdir=runs
   ```
    **Optional: Visualize imagine sequences**: Using saved models to visualize agent's prediction of environment dynamic. You would need to create a config folder in the run logging directory and drag the training config file in
   ```
   python imagine.py --runpath <Path to run file>
   ```

## Citation
This implementation was made possible thanks to these papers.
```bibtex
@article{hafner2019dream,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
@misc{1801.00690,
  title = {DeepMind Control Suite},
  author = {Yuval Tassa and Yotam Doron and Alistair Muldal and Tom Erez and Yazhe Li and Diego de Las Casas and David Budden and Abbas Abdolmaleki and Josh Merel and Andrew Lefrancq and Timothy Lillicrap and Martin Riedmiller},
  journal = {arXiv preprint arXiv:1801.00690},
  year = {2018},
}

```

## Contributions

Contributions are welcome! Whether it's extending functionality, improving efficiency, or correcting bugs, your input helps make this project better for everyone.
