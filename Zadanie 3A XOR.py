import numpy as np
import matplotlib.pyplot as plt

# Seed pre replikovanie výsledkov
#np.random.seed(123)

# Lineárna vrstva
class Linear:
    def __init__(self, input_dim, output_dim):
        # Inicializácia váh náhodnými hodnotami z intervalu [-1, 1]
        self.weights = np.random.uniform(-1, 1, (input_dim, output_dim))
        
        # Inicializácia biasov na nuly
        self.biases = np.zeros((1, output_dim))
        
        # Uchováva vstupnú hodnotu pre spätný prechod
        self.input = None
        
        # Premenné na ukladanie gradientov váh a biasov počas spätného prechodu
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x):
        # Uloženie vstupu pre neskorší spätný prechod
        self.input = x
        
        # Výpočet výstupu vrstvy
        # Vracia lineárnu kombináciu: y = xW + b
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output):
        # Gradient váh: x^T * grad_output
        self.grad_weights = np.dot(self.input.T, grad_output)
        
        # Gradient biasov: súčet všetkých hodnôt v grad_output po batchi
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient pre predchádzajúcu vrstvu: grad_output * W^T
        return np.dot(grad_output, self.weights.T)

    def update_params(self, lr, momentum, grad_weights_prev, grad_biases_prev):
        # Výpočet zmeny váh pomocou gradientov a momenta
        update_weights = momentum * grad_weights_prev - lr * self.grad_weights
        update_biases = momentum * grad_biases_prev - lr * self.grad_biases

        # Aktualizácia váh a biasov vrstvy
        self.weights += update_weights
        self.biases += update_biases

        # Vráti nové zmeny pre použitie v ďalšej iterácii
        return update_weights, update_biases

# Sigmoid funkcia aktivácie
class Sigmoid:
    def __init__(self):
        # Uchováva výstup sigmoid funkcie z dopredného prechodu
        self.output = None

    def forward(self, x):
        # Výpočet sigmoid funkcie
        self.output = 1 / (1 + np.exp(-x))
        
        # Výstup transformovaný sigmoid funkciou
        return self.output

    def backward(self, grad_output):
        # Výpočet gradientu sigmoid funkcie
        sigmoid_grad = self.output * (1 - self.output)
        
        # Prenos gradientu späť
        return grad_output * sigmoid_grad

# ReLU funkcia aktivácie
class ReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha  # Koeficient pre záporné hodnoty (leaky ReLU)
        self.output = None  # Uchováva výstup ReLU funkcie pre spätný prechod

    def forward(self, x):
        # Výpočet výstupu ReLU funkcie (parametrická verzia)
        self.output = np.where(x > 0, x, self.alpha * x)
        
        # Vracia transformovaný výstup
        return self.output

    def backward(self, grad_output):
        # Výpočet gradientu ReLU: 1 pre kladné hodnoty, alpha pre záporné
        grad = np.where(self.output > 0, 1, self.alpha)
        
        # Prenos gradientu späť
        return grad_output * grad

# Tanh funkcia aktivácie
class Tanh:
    def __init__(self):
        self.output = None  # Uchováva výstup tanh funkcie z dopredného prechodu

    def forward(self, x):
        # Výpočet Tanh funkcie
        self.output = np.tanh(x)
        
        # Vracia transformovaný výstup
        return self.output

    def backward(self, grad_output):
        # Výpočet gradientu Tanh: 1 - (tanh(x))^2
        tanh_grad = 1 - self.output ** 2
        
        # Prenos gradientu späť
        return grad_output * tanh_grad

# MSE chybová funkcia
class MSE:
    def __init__(self):
        self.y_true = None  # Skutočné hodnoty
        self.y_pred = None  # Predikované hodnoty

    def forward(self, y_true, y_pred):
        # Výpočet priemernej štvorcovej chyby
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        # Gradient MSE chyby
        return -2 * (self.y_true - self.y_pred) / self.y_true.shape[0]

# Model neurónovej siete
class Model:
    def __init__(self, layers):
        # Zoznam vrstiev a inicializácia momentum pre váhy a biasy
        self.layers = layers
        self.momentum_weights = [np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None for layer in layers]
        self.momentum_biases = [np.zeros_like(layer.biases) if hasattr(layer, 'biases') else None for layer in layers]

    def forward(self, x):
        # Prechod dát sieťou (forward pass)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        # Spätné šírenie gradientu cez vrstvy
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def update_params(self, lr, momentum):
        # Aktualizácia parametrov vo všetkých vrstvách
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                self.momentum_weights[i], self.momentum_biases[i] = layer.update_params(
                    lr, momentum, self.momentum_weights[i], self.momentum_biases[i]
                )

# Funkcia na trénovanie a testovanie
def train_and_test(X, y, epochs, lr, momentum, model, criterion, problem_name):
    losses = []  # Zoznam na ukladanie strát
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)
        # Výpočet chyby
        loss = criterion.forward(y, y_pred)
        losses.append(loss)
        # Spätné šírenie
        loss_grad = criterion.backward()
        model.backward(loss_grad)
        # Aktualizácia parametrov
        model.update_params(lr, momentum)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Vykreslenie grafu chyby
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses, label=f'{problem_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {problem_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predikcia po trénovaní
    y_pred = model.forward(X)
    y_pred_rounded = np.round(y_pred)  # Zaokrúhlenie predikcií
    y_pred_fixed = np.where(y_pred_rounded == -0., 0., y_pred_rounded)  # Oprava hodnôt -0
    print("Predictions:", y_pred_fixed)

# Architektúra siete
layers = [
    Linear(2, 4),  
    ReLU(),
    #Linear(4, 4),
    #ReLU(),         
    Linear(4, 1),  
    ReLU()         
]

# Inicializácia modelu a chyby
model = Model(layers)
criterion = MSE()

# Trénovanie na logické problémy
print("XOR Problem:")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])
train_and_test(X_xor, y_xor, epochs=500, lr=0.1, momentum=0.01, model=model, criterion=criterion, problem_name="XOR")

print("AND Problem:")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])
train_and_test(X_and, y_and, epochs=500, lr=0.3, momentum=0.01, model=model, criterion=criterion, problem_name="AND")

print("OR Problem:")
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
train_and_test(X_or, y_or, epochs=500, lr=0.3, momentum=0.01, model=model, criterion=criterion, problem_name="OR")
