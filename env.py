from typing import Dict, Tuple
import gym
from gym import spaces
import numpy as np
import random
from pyBKT.models import Model
import pickle


emotions = ["anger", "surprise", "disgust", "enjoyment", "fear", "sadness"]

categories = ["literature", "vocabulary", "idioms", "grammar"]


class StudentEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self):
        # Define action and observation space
        # Action space: Categories of questions to ask
        self.action_space = spaces.Discrete(len(categories))

        # Observation space: Emotions as states
        self.observation_space = spaces.Discrete(len(emotions))

        # Define initial state
        self.current_emotion = self.np_random.choice(len(emotions))

        # Load trained knowledge model
        self.knowledge_model = Model(seed=42)
        self.knowledge_model.load("knowledge.pkl")

        # Load trained emotions model
        self.emotions_model = None

        with open("emotions.pkl", "rb") as file:
            self.emotions_model = pickle.load(file)

    def _get_obs(self):
        return self.current_emotion

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed, options=options)

        self.current_emotion = self.np_random.choice(len(emotions))

        return self._get_obs(), {}

    def step(self, action):
        # Execute one time step within the environment

        # Apply stochastic rules to determine correctness and new emotion
        correct, next_emotion = self.apply_rules(action)

        # Update the current emotion based on the rules or any other logic
        self.current_emotion = emotions.index(next_emotion)

        # Return the new state, reward, and other info
        return self._get_obs(), correct, True, True, {}

    def apply_rules(self, category: str) -> Tuple[int, str]:
        emotion = emotions[self.current_emotion]

        rule: Dict[Tuple[str, str], dict] = rules.get((emotion, category), None)
        if rule is not None:
            correct_chance: float = rule["correct_chance"]
            next_emotion_probs: Dict[str, float] = rule["next_emotion"]

            # Determine correctness based on the chance
            correct = 10 if random.random() < correct_chance else -5

            # Determine the next emotion based on probabilities
            next_emotion = random.choices(
                list(next_emotion_probs.keys()),
                weights=list(next_emotion_probs.values()),
            )[0]

            return correct, next_emotion
        else:
            # Default case if there's no specific rule defined
            return random.choice([10, -5]), random.choice(emotions)

    def render(self):
        pass

    def close(self):
        pass
