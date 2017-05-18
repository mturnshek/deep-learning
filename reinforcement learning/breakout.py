from dqn import learning
from dqn import models

agent = learning.Agent('Breakout-v0', models.conv)
agent.play()