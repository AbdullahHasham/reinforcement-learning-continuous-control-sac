# Deep Reinforcement Learning for Continuous Control

**Comparative Study of Reward Engineering and Policy Optimization in MountainCarContinuous-v0**

---

## ⚖️ Executive Summary
This report analyzes policy optimization in the continuous control domain using the **MountainCarContinuous-v0** environment. We compare the sample efficiency and convergence dynamics of **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)**. Due to the default environment's sparse reward (+100 at the goal, penalized by active force costs), standard off-policy baseline models fail to solve the task (0% success rate). To address this, we implement **Potential-Based Reward Shaping (PBRS)**, which combines height potential, kinetic energy bonuses, and goal proximity indices. Under this engineered reward function, the SAC agent successfully solves the environment, achieving a **100% success rate** in its best trial with an average episode length of only **69.9 steps**, significantly outperforming PPO.

---

## 1. Problem Formulation & Environment Specifications

The **MountainCarContinuous-v0** environment presents an underpowered car positioned in a valley between two hills. The goal is to drive the car up the right hill to reach the flag. Because the engine is weaker than gravity, the agent cannot drive directly up the hill; it must build momentum by rocking back and forth.

### 📊 Mathematical Specifications
*   **State Space ($\mathcal{S}$):** Continuous 2D vector $\mathbf{s}_t = [p_t, v_t]^T$:
    *   *Car Position ($p$):* $[-1.2, 0.6]$ (Goal flag is located at $p \ge 0.45$).
    *   *Car Velocity ($v$):* $[-0.07, 0.07]$.
*   **Action Space ($\mathcal{A}$):** Continuous motor acceleration force $a_t \in [-1, 1]$.
*   **Transition Dynamics:** 
    $$v_{t+1} = v_t + 0.0015 \cdot a_t - 0.0025 \cdot \cos(3 \cdot p_t)$$
    $$p_{t+1} = p_t + v_{t+1}$$
*   **Default Environment Reward ($R_{default}$):**
    $$R(s_t, a_t) = \begin{cases} 
      +100 & \text{if } p_t \ge 0.45 \\
      -0.1 \cdot a_t^2 & \text{otherwise}
   \end{cases}$$
   *This reward is highly sparse, penalizing exploration unless the agent reaches the goal.*

---

## 2. Reward Engineering & Potential-Based Shaping (PBRS)

To resolve the exploration bottleneck of sparse feedback, we implement **Potential-Based Reward Shaping (PBRS)**. PBRS is mathematically guaranteed to preserve the optimal policy ($\pi^*$) of the original MDP while providing dense feedback at every timestep.

The shaping function is formulated as:
$$F(s_t, a_t, s_{t+1}) = \gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)$$

The potential function $\Phi(s)$ is designed as a weighted linear combination of three oenological physical indicators:
1.  **Height Potential ($\Phi_{height}$):** Measures gravitational potential energy based on the valley terrain:
    $$\Phi_{height}(s) = \sin(3 \cdot p) \cdot 0.45 + 0.55$$
2.  **Kinetic Energy Potential ($\Phi_{kinetic}$):** Measures the momentum built by the vehicle to encourage swing behavior:
    $$\Phi_{kinetic}(s) = 0.5 \cdot v^2$$
3.  **Goal Proximity Potential ($\Phi_{proximity}$):** Scales with distance to the target position:
    $$\Phi_{proximity}(s) = 1.0 - \frac{|p_{goal} - p|}{\text{range}}$$

---

## 3. Deep Reinforcement Learning Algorithms

We evaluate two distinct DRL architectures:

### 3.1 Soft Actor-Critic (SAC)
An off-policy, actor-critic algorithm utilizing maximum entropy reinforcement learning. SAC optimizes a trade-off between expected return and entropy (exploration rate):
$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}} \left[ R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
*   *Parameters:* Learning rate ($3 \cdot 10^{-4}$), buffer size ($10^5$), batch size ($256$), $\gamma = 0.99$.

### 3.2 Proximal Policy Optimization (PPO)
An on-policy, policy gradient algorithm that constrains policy updates using a clipped objective function to maintain training stability:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
*   *Parameters:* Learning rate ($1 \cdot 10^{-4}$), steps per rollout ($2048$), batch size ($64$), $\gamma = 0.99$.

---

## 4. Empirical Evaluation & Convergence Dynamics

### 4.1 Baseline Performance (Default Sparse Reward)
The SAC baseline agent trained on $R_{default}$ completely failed to solve the task, achieving **0% success** across all three seeds. Without dense potential guidance, the agent learned to output near-zero actions to avoid control penalties, converging to a local optimum of "doing nothing" (Episode Length: $999.0$).

### 4.2 Custom Shaped Performance (SAC vs PPO)
Retraining with the potential-based wrapper yielded major performance improvements:

| Algorithm & Config | Trial 1 Reward | Trial 2 Reward | Trial 3 Reward | Success Rate (%) | Avg Episode Length |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SAC (Default Baseline)** | -0.00003 | -0.0002 | -0.0005 | 0.0% | 999.0 |
| **SAC (Custom Shaped)** | -0.004 | 94.76 | -0.006 | **33.3%** | **689.3 (Best: 69.9)** |
| **PPO (Custom Shaped)** | -0.0004 | -0.0004 | -0.0004 | 0.0% | 999.0 |

### 🔍 Interpretations & Findings
1.  **SAC Efficiency:** Under seed 2, the custom-shaped SAC model achieved a 100% success rate, completing the climb in **69.9 steps**. However, high variance across trials (33.3% average success) indicates high stochastic sensitivity to early random explorations.
2.  **PPO Limitations:** PPO failed to achieve goal-reaching behaviors. Although tensorboard logs showed steady optimization of the shaped rewards (height and kinetic potentials), PPO failed to bridge the training-to-evaluation gap, getting trapped in local maxima.
3.  **Exploration Strategy:** SAC’s maximum-entropy policy ($\mathcal{H}$) proved superior in continuous spaces by promoting wider exploration, whereas PPO's clipped gradients restricted policy updates, causing premature convergence.

---

## 5. Conclusion & Future Work
This project demonstrates that **potential-based reward shaping** is crucial for solving sparse continuous control tasks. The off-policy SAC model utilizing maximum entropy achieved successful policies, whereas the on-policy PPO fell short. 

Future extensions will incorporate **Hindsight Experience Replay (HER)** to learn from failed attempts, alongside hyperparameter tuning of the entropy coefficient ($\alpha$) to stabilize SAC variance across different random seeds.

---

## 👥 Authors & Contributors
*   **Hasham Abdullah** - *Deep RL Engineer & Developer*
*   **Souhaib Othmani** - *Deep RL Engineer & Analyst*
