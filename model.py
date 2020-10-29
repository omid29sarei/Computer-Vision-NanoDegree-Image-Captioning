import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,
                            num_layers,batch_first=True,dropout=0.2)
        self.fc1 = nn.Linear(hidden_size,vocab_size)
        
    
    def forward(self, features, captions):
        emb = self.word_embed(captions[:,:-1])
        emb = torch.cat((features.unsqueeze(1),emb),1)
        lstm_hidden,_ = self.lstm(emb)
        output = self.fc1(lstm_hidden)
        return output
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_list_ids = []

        for i in range(20):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc1(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_list_ids.append(predicted.tolist()[0])
            inputs = self.word_embed(predicted)
            inputs = inputs.unsqueeze(1)          

        return sampled_list_ids
            