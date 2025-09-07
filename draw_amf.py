import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

def draw_amf(amf_file):
    amf = torch.load(amf_file)
    