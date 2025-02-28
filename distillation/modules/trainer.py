import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

from .student import StudentModel
from .teacher import TeacherModel
from .data_module import DistillationDataModule


class DistillationModel(pl.LightningModule):
    def __init__(self, student_model, teacher_model, learning_rate=1e-3):
        super(DistillationModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Weight for distillation loss
        self.alpha = 0.5
            
    def train_loss(self, distillation_loss, classification_loss):
        return self.alpha * distillation_loss + (1 - self.alpha) * classification_loss
        

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        student_output = self.student_model(data)
        
        # Get teacher output with no gradient calculation
        with torch.no_grad():
            teacher_output = self.teacher_model(data)
        
        # Calculate distillation loss
        temperature = 2.0  # You can adjust this temperature
        distillation_loss = self.loss_fn(
            nn.functional.log_softmax(student_output / temperature, dim=-1),
            nn.functional.softmax(teacher_output / temperature, dim=-1)
        ) * (temperature ** 2)
        
        # Calculate standard classification loss
        classification_loss = self.classification_loss(student_output, targets)
        
        # Combine the losses
        
        train_loss = self._train_loss(distillation_loss, classification_loss)
        
        self.log('train_loss', train_loss)
        self.log('distillation_loss', distillation_loss)
        self.log('classification_loss', classification_loss)
        
        return train_loss

    def configure_optimizers(self):
        return Adam(self.student_model.parameters(), lr=self.learning_rate)

def train_distillation(batch_size=32, max_epochs=10):
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        batch_size=batch_size,
        strategy='deepspeed',
        precision=16,
        accumulate_grad_batches=4)
    trainer.fit(
        model=DistillationModel(
            student_model=StudentModel(input_dim=784, hidden_dim=16, output_dim=10), 
            teacher_model=TeacherModel(input_dim=784, hidden_dim=128, output_dim=10)),
        datamodule=DistillationDataModule(
            train_data="path/to/train_data",
            val_data="path/to/val_data",
            test_data="path/to/test_data",
            batch_size=batch_size))
