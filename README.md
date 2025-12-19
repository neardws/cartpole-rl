# CartPole 强化学习

使用 Deep Q-Network (DQN) 算法解决 OpenAI Gymnasium 的 CartPole-v1 平衡木问题。

## 环境说明

CartPole 问题：通过左右移动小车来保持杆子平衡。

- **状态空间**: 4维连续空间 (位置, 速度, 角度, 角速度)
- **动作空间**: 2个离散动作 (左移, 右移)
- **奖励**: 每个时间步 +1
- **终止条件**: 杆子倾斜超过15度 或 小车移出边界 或 达到500步

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python train.py
```

### 评估

```bash
python evaluate.py
```

## 算法

使用 DQN (Deep Q-Network) 算法，主要特点：

- Experience Replay (经验回放)
- Target Network (目标网络)
- Epsilon-greedy 探索策略

## 项目结构

```
├── agent.py          # DQN Agent 实现
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── requirements.txt  # 依赖
└── README.md         # 说明文档
```

## 参考

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
