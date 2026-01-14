import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=4):
        """
        Definirea arhitecturii Rețelei Neuronale.
        Input: 63 features (21 puncte * 3 coordonate x,y,z)
        Output: 4 clase (STOP, INAINTE, STANGA, DREAPTA)
        """
        super(GestureClassifier, self).__init__()
        
        # Stratul 1: Intrare -> Hidden Layer 1
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2) # Previne overfitting
        
        # Stratul 2: Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Stratul 3: Hidden Layer 2 -> Output Layer
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Fluxul de date prin rețea (Forward pass)
        """
        # Activare ReLU pentru straturile ascunse
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Stratul final returnează logits (valori brute)
        # Softmax se aplică de obicei la calculul erorii sau la inferență
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Test rapid al arhitecturii
    model = GestureClassifier()
    print(model)
    # Test cu un input random
    dummy_input = torch.randn(1, 63)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Trebuie să fie [1, 4]