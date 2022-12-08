import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet50(pretrained=True)     # deprecated
        weights = models.ResNet50_Weights.DEFAULT       # models.ResNet50_Weights: pretrained weights for `models.resnet50`
        resnet  = models.resnet50( weights = weights )
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules      = list(resnet.children())[:-1]
        self.resnet  = nn.Sequential(*modules)
        self.embed   = nn.Linear(resnet.fc.in_features, embed_size)
        
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        features = self.relu(features)
        features = self.dropout(features)
        
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed   = nn.Embedding(vocab_size, embed_size)
        self.lstm    = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear  = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        
        embeddings = self.embed( captions[:,:-1] ) # Removing last tokens from captions making dimension size 1 less
        embeddings = self.dropout( embeddings )
        embeddings = torch.cat( (features.unsqueeze(1), embeddings), dim=1 )
        hiddens, _ = self.lstm(embeddings)
        outputs    = self.linear(hiddens)
        
        return outputs
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
















































