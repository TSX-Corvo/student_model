from typing import Dict, Tuple
import gym
from gym import spaces
import numpy as np
import random
from pyBKT.models import Model
import pickle
import pandas as pd


emotions = ["Ira", "Sorpresa", "Disgusto", "Disfrute", "Miedo", "Tristeza"]

categories = [
    "Ortografía y Gramática",
    "Elementos Narrativos",
    "Gramática y Sintaxis",
    "Comunicación y Lenguaje",
    "Figuras Literarias",
]

# Question ids for each skill, generated like this because the data is sorted by skill beforehand
questions_by_category = {
    idx: list(range((idx * 6) + 1, (idx + 1) * 6)) for idx, cat in enumerate(categories)
}


column_labels = ["student_id", "question_id", "skill", "correct"]

prior_data = pd.read_csv("train_db.csv")


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
        # Initialize simulated interaction

        # print(category)

        student_id: int = 1  # fixed because of data
        question_id = np.random.randint(
            questions_by_category[category][0],
            questions_by_category[category][-1],
        )  # random question based on category
        skill = category

        prior_correct = prior_data.loc[prior_data["question_id"] == question_id][
            "correct"
        ].values[
            0
        ]  # Recall prior knowledge

        # Predict the correctness

        obs = [student_id, question_id, skill, prior_correct]

        temp = pd.DataFrame(
            data=np.array([obs]),
            columns=column_labels,
        )

        try:
            prediction = self.knowledge_model.predict(data=temp)
        except:
            prediction = {"correct_predictions": [0]}

        correct_chance = prediction["correct_predictions"][0]

        correct = 10 if random.random() < correct_chance else -5

        # Predict the emotion

        next_emotion_val = self.emotions_model.predict([obs])[0]

        next_emotion = emotions[next_emotion_val]

        # Return (correct, emotion) pair
        return correct, next_emotion

    def render(self):
        pass

    def close(self):
        pass
