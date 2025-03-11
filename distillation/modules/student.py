import torch

import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def distillation_loss(self, student_output, teacher_output, temperature=2.0):
        student_log_probs = F.log_softmax(student_output / temperature, dim=1)
        teacher_probs = F.softmax(teacher_output / temperature, dim=1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        return loss
