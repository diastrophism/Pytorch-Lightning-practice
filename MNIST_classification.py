import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import functional as FM

class Classifier(pl.LightModule):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Flatten(),
			nn.Linear(28*28, 64),
			nn.BatchNorm1d(512),
			
