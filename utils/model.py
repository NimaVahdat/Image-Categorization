import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

import numpy as np

from SpykeTorch import snn
from SpykeTorch import functional as sf

import sys

class DeepSNN(nn.Module):
	def __init__(self, prop):
		super(DeepSNN, self).__init__()
		in_channel1 = prop["in_channel1"]
		in_channel2 = prop["in_channel2"]
		out_channel = prop["out_channel"]
		weight_mean = prop["weight_mean"]
		weight_std = prop["weight_std"]
		lr = prop["lr"]
		k1 = prop["k1"]
		k2 = prop["k2"]
		r1 = prop["r1"]
		r2 = prop["r2"]
        
		self.conv1 = snn.Convolution(in_channel1,
                                     in_channel2,
                                     5,
                                     weight_mean=weight_mean,
                                     weight_std=weight_std)
		self.conv1_t = 10
		self.k1 = k1
		self.r1 = r1

		self.conv2 = snn.Convolution(in_channel2,
                                     out_channel,
                                     2,
                                     weight_mean=weight_mean,
                                     weight_std=weight_std)
		self.conv2_t = 1
		self.k2 = k2
		self.r2 = r2

		self.stdp1 = snn.STDP(self.conv1, lr)
		self.stdp2 = snn.STDP(self.conv2, lr)
		self.max_ap = Parameter(torch.Tensor([0.15]))

		self.ctx = {"input_spikes":None,
                    "potentials":None,
                    "output_spikes":None,
                    "winners":None}
		self.spike_counter = 0
	
	def save_data(self, input_spike, potentials, output_spikes, winners):
		self.ctx["input_spikes"] = input_spike
		self.ctx["potentials"] = potentials
		self.ctx["output_spikes"] = output_spikes
		self.ctx["winners"] = winners

	def forward(self, input, layer_idx):
		input = sf.pad(input.float(), (2,2,2,2))
		if self.training:
            # Calculating potentials
			v = self.conv1(input)
            # Spikes and Potentails
			s, v = sf.fire(v, self.conv1_t, True)

			if layer_idx == 1:
                # Updating learning rule
				self.spike_counter += 1
				if self.spike_counter >= 500:
					self.spike_counter = 0
					ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75
					self.stdp1.update_all_learning_rate(ap.item(), an.item())
                 
				v = sf.pointwise_inhibition(v)
				s = v.sign()
				winners = sf.get_k_winners(v, self.k1, self.r1, s)
				self.save_data(input, v, s, winners)
				return s, v
            
			s_in = sf.pad(sf.pooling(s, 2, 2, 1), (1,1,1,1))
			s_in = sf.pointwise_inhibition(s_in)
			v = self.conv2(s_in)
			s, v = sf.fire(v, self.conv2_t, True)
            
			if layer_idx == 2:
				v = sf.pointwise_inhibition(v)
				s = v.sign()
				winners = sf.get_k_winners(v, self.k2, self.r2, s)
				self.save_data(s_in, v, s, winners)
				return s, v
			s_out = sf.pooling(s, 2, 2, 1)
			return s_out
		else:
			v = self.conv1(input)
			s, v = sf.fire(v, self.conv1_t, True)
			v = self.conv2(sf.pad(sf.pooling(s, 2, 2, 1), (1,1,1,1)))
			s, v = sf.fire(v, self.conv2_t, True)
			s = sf.pooling(s, 2, 2, 1)
			return s
	
	def stdp(self, layer_idx):
		if layer_idx == 1:
			self.stdp1(self.ctx["input_spikes"],
                       self.ctx["potentials"], 
                       self.ctx["output_spikes"], 
                       self.ctx["winners"])
		if layer_idx == 2:
			self.stdp2(self.ctx["input_spikes"],
                       self.ctx["potentials"],
                       self.ctx["output_spikes"], 
                       self.ctx["winners"])
    
	def train_model(self, data, layer_idx):
		self.train()
		for i in range(len(data)):
			sys.stdout.write("\rIteration ----->  %i" % i)
			sys.stdout.flush()
			data_in = data[i]
			self(data_in, layer_idx)
			self.stdp(layer_idx)

	def test(self, data, target, layer_idx):
		self.eval()
		ans = [None] * len(data)
		t = [None] * len(data)
		for i in range(len(data)):
			data_in = data[i]
			output,_ = self(data_in, layer_idx).max(dim = 0)
			ans[i] = output.reshape(-1).cpu().numpy()
			t[i] = target[i]
		return np.array(ans), np.array(t) 