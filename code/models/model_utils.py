import torch
import torch.nn as nn

def gaussian_basis(
    distance, max_distance, num_centers, width, min_distance=0, to_device=True
):
    '''
    creates soft binning of distance
    '''
    centers = torch.linspace(min_distance, max_distance, num_centers)
    
    if to_device:
        centers = centers.to(distance.device)
    
    # in case identifier for virutal edges is present
    positions = centers - distance
    gaussian_expansion = torch.exp(-positions * positions / (width * width))
    
    return gaussian_expansion


class EmbeddingLayer(nn.Module):
    
    def __init__(self, in_features, emb_size):
        
        super().__init__()
        
        self.emb = nn.Linear(in_features, emb_size)
        
    def forward(self, x):
        
        return self.emb(x)