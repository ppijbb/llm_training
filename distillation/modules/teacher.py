import torch

import torch.nn as nn
import torch.optim as optim
from transformers import Phi3ForCausalLM, AutoTokenizer

class TeacherModel(nn.Module):
    def __init__(self,):
        super(TeacherModel, self).__init__()
        self.teacher_name = "microsoft/phi-4"
        self.teacher = Phi3ForCausalLM.from_pretrained(
            self.teacher_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_name
        )
        
    @torch.inference_mode
    def forward(self, x):
        return self.teacher(x)

def train_teacher_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage:
# model = TeacherModel(input_dim=784, hidden_dim=128, output_dim=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_teacher_model(model, dataloader, criterion, optimizer, num_epochs=20)

test = TeacherModel()
print(test.teacher.config)
