import pickle
import pytest
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class Data(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float


def load_models():
    model = pickle.load(open(os.path.join(os.getcwd(),
                                          '\\models\\model.pkl', 'rb')))
    scaler = pickle.load(open(os.path.join(os.getcwd(),
                                           '\\models\\scaler.pkl', 'rb')))
    return scaler, model
import os
print(os.path.join(os.getcwd()))
scaler, model = load_models()