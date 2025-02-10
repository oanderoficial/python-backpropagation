import numpy as np

# Função de ativação Sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivada da sigmoid

# Dados de entrada (4 exemplos, 2 características)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Saída esperada
y = np.array([[0], [1], [1], [0]])  # Problema do XOR


# Inicializa pesos aleatórios
np.random.seed(42)
pesos_entrada_oculta = np.random.uniform(-1, 1, (2, 5))  # 2 entradas -> 5 neurônios ocultos
pesos_oculta_saida = np.random.uniform(-1, 1, (5, 1))    # 5 neurônios ocultos -> 1 saída


# Taxa de aprendizado
learning_rate = 0.3

# Treinamento por 10.000 iterações
for epoch in range(20000):
    # Forward Pass
    camada_oculta = sigmoid(np.dot(X, pesos_entrada_oculta))  # Entrada -> Oculta
    camada_saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida))  # Oculta -> Saída

    # Erro
    erro = y - camada_saida

    # Backpropagation
    d_saida = erro * sigmoid_derivative(camada_saida)  # Gradiente da saída
    erro_oculta = d_saida.dot(pesos_oculta_saida.T)  # Propagação do erro para a camada oculta
    d_oculta = erro_oculta * sigmoid_derivative(camada_oculta)  # Gradiente da oculta

    # Atualização dos pesos
    pesos_oculta_saida += camada_oculta.T.dot(d_saida) * learning_rate
    pesos_entrada_oculta += X.T.dot(d_oculta) * learning_rate

    # Exibir erro a cada 1000 iterações
    if epoch % 1000 == 0:
        print(f"Erro na época {epoch}: {np.mean(np.abs(erro)):.4f}")

# Teste após treinamento
print("\nSaída final da rede após treinamento:")
print(camada_saida)
