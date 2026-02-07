from typing import Dict
from src.framework.base_agent import BaseAgent

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self.agents[agent.name] = agent

    def start_all(self):
        for name, agent in self.agents.items():
            agent.start()

    def stop_all(self):
        for name, agent in self.agents.items():
            agent.stop()
