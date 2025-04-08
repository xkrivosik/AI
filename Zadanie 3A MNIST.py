import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Načítanie a predspracovanie datasetu
transform = Compose([
    ToTensor(),  # Konvertuje obrázky na tenzory a škáluje hodnoty do intervalu [0, 1]
])

# Načítanie MNIST datasetu
train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

# DataLoaders na dávkovanie tréningových a testovacích dát
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Definícia viacvrstvového perceptrónu
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)      
        self.fc4 = nn.Linear(64, 10)       
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU alebo CPU podľa dostupnosti

# Tréningová funkcia
def train(model, loader, optimizer, criterion):
    model.train()  # Nastavenie modelu do tréningového módu
    total_loss = 0
    correct = 0
    for data, target in loader:  # Prechod cez dávky dát
        data, target = data.to(device), target.to(device)  # Prenos na zariadenie (GPU/CPU)
        optimizer.zero_grad()  # Vynulovanie gradientov
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Výpočet chyby
        loss.backward()  # Spätná propagácia
        optimizer.step()  # Aktualizácia váh
        total_loss += loss.item()  # Akumulácia straty
        pred = output.argmax(dim=1)  # Predikcia: trieda s najvyššou pravdepodobnosťou
        correct += pred.eq(target).sum().item()  # Počet správnych predikcií
    accuracy = 100. * correct / len(loader.dataset)  # Výpočet presnosti
    return total_loss / len(loader), accuracy  # Priemerná strata a presnosť

# Testovacia funkcia
def test(model, loader, criterion):
    model.eval()  # Nastavenie modelu do evaluačného módu
    total_loss = 0
    correct = 0
    with torch.no_grad():  # Deaktivácia výpočtu gradientov
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Výpočet chyby
            total_loss += loss.item()  # Akumulácia straty
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Optimalizačné algoritmy
optimizers = {
    "SGD": lambda model: optim.SGD(model.parameters(), lr=0.01),
    "SGD_momentum": lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    "Adam": lambda model: optim.Adam(model.parameters(), lr=0.001)
}

results = {}  # Na uloženie výsledkov pre každý optimalizátor

# Tréning a testovanie modelu pre každý optimalizátor
for opt_name, opt_fn in optimizers.items():
    print(f"Training with {opt_name}...")
    model = MLP().to(device)  # Inicializácia modelu a prenos na zariadenie
    optimizer = opt_fn(model)  # Výber optimalizátora
    criterion = nn.CrossEntropyLoss()  # Funkcia na výpočet chyby

    # Na uloženie priebežných výsledkov
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    num_epoch = 10
    for epoch in range(num_epoch):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, test_loader, criterion)

        # Ukladanie výsledkov
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Výpis výsledkov pre aktuálnu epochu
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Uloženie výsledkov pre aktuálny optimalizátor
    results[opt_name] = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }

# Výpočet a vizualizácia confusion matrix
def plot_confusion_matrix(model, loader):
    model.eval()  # Nastavenie modelu do evaluačného módu
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Vytvorenie confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # Vizualizácia
    plt.title("Confusion Matrix")
    plt.show()

# Nájdeme najlepší model na základe presnosti
best_optimizer = max(results.items(), key=lambda x: max(x[1]["test_accuracies"]))[0]
print(f"Best optimizer: {best_optimizer}")

# Inicializácia a trénovanie modelu s najlepším optimalizátorom
best_model = MLP().to(device)
best_optimizer_fn = optimizers[best_optimizer]
optimizer = best_optimizer_fn(best_model)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    train_loss, train_acc = train(best_model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(best_model, test_loader, criterion)

# Zobrazenie confusion matrix pre najlepší model
plot_confusion_matrix(best_model, test_loader)

# Vizualizácia výsledkov
for opt_name, result in results.items():
    plt.plot(result["test_accuracies"], label=opt_name)  # Graf presnosti na testovacej množine

plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
