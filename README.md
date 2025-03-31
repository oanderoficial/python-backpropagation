# Python-backpropagation

Exemplo básico de backpropagation em Python usando NumPy

* Usando backpropagation para treinar uma rede neural simples a resolver o problema XOR. A rede aprende ajustando os pesos automaticamente com base no erro.

## Funções de ativação: Sigmoide vs Tanh

![image](https://github.com/user-attachments/assets/49d9be2f-8c1f-4614-b8d7-d325d8b976a6)

## O problema XOR
![image](https://github.com/user-attachments/assets/9863ebb3-07ba-43a8-90d5-104701d08365)

O problema de XOR (ou OU exclusivo) é um problema clássico em redes neurais e aprendizado de máquina. Ele se refere à tentativa de treinar um perceptron (um modelo de rede neural simples) para aprender a função XOR.

## Tabela 

![image](https://github.com/user-attachments/assets/0e25d1ba-0417-42eb-bce0-b72253e1757c)


## A lógica do XOR é simples:

* Retorna 1 se apenas uma das entradas for 1.
* Retorna 0 caso contrário.

## Passos 

1) Inicializa os pesos e os dados de entrada.
2) Passagem direta (forward pass) para calcular a saída.
3) Cálculo do erro e retropropagação do gradiente para ajustar os pesos.
4) Atualização dos pesos usando o gradiente descendente.

## 1) importe a biblioteca numpy 

```python
import numpy as np
```

*  Será usado para operações matemáticas, como multiplicação de matrizes e cálculo da função de ativação.

## 2) Definição da Função de Ativação Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
* A função sigmoid é uma função de ativação usada para transformar um valor numérico em um número entre 0 e 1.
* Isso é útil porque podemos interpretar esse valor como uma probabilidade.

## Exemplo: 

```bash
sigmoid(0)  # Resultado: 0.5
sigmoid(2)  # Resultado: 0.88 (próximo de 1)
sigmoid(-2) # Resultado: 0.12 (próximo de 0)
````

## 3) Derivada da Função Sigmoid

```python
def sigmoid_derivative(x):
    return x * (1 - x)
```
* A derivada da função sigmoid é usada no backpropagation para calcular os ajustes necessários nos pesos.
* A fórmula vem da regra da derivação da função sigmoide.

## Exemplo 

```bash
x = sigmoid(2)  # 0.88
sigmoid_derivative(x)  # 0.105 (menor quando x está próximo de 0 ou 1)
```

## 4) Criando os Dados de Entrada (X) e Saída (y)

```python
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # Problema do XOR
```

* X é a matriz de entrada com 4 exemplos (linhas) e 2 características (colunas).
* y é a saída esperada, baseada na tabela verdade do XOR.

## Interpretação:

* A entrada (0,0) deve produzir saída 0
* A entrada (0,1) deve produzir saída 1
* A entrada (1,0) deve produzir saída 1
* A entrada (1,1) deve produzir saída 0 (operação XOR)

## 5) Inicialização dos Pesos

```python
np.random.seed(42)  # Para garantir resultados reproduzíveis
pesos_entrada_oculta = np.random.uniform(-1, 1, (2, 5))  # 2 entradas -> 5 neurônios ocultos
pesos_oculta_saida = np.random.uniform(-1, 1, (5, 1))    # 5 neurônios ocultos -> 1 saída
```

* Os pesos das conexões são inicializados aleatoriamente no intervalo [-1, 1].
* pesos_entrada_oculta: Matriz (2x5) (2 neurônios de entrada → 5 neurônios na camada oculta)
* pesos_oculta_saida: Matriz (5x1) (5 neurônios da camada oculta → 1 neurônio na saída)

## Exemplo de pesos gerados:

```python
pesos_entrada_oculta =
[[-0.25, 0.45, -0.12],
 [0.78, -0.56, 0.34]]

pesos_oculta_saida =
[[-0.68],
 [0.22],
 [-0.89]]
```

## 6) Definição da Taxa de Aprendizado

```python
learning_rate = 0.3
```
* Taxa de aprendizado controla o tamanho dos ajustes feitos nos pesos durante o treinamento.
* Valores muito grandes podem fazer o treinamento divergir, e valores muito pequenos podem torná-lo muito lento.


## 7) Loop de Treinamento

```python
for epoch in range(20000):
```
* O modelo será treinado por 20.000 iterações.

## Forward Pass (Passagem Direta)

```python
camada_oculta = sigmoid(np.dot(X, pesos_entrada_oculta))  # Entrada -> Oculta
camada_saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida))  # Oculta -> Saída
```

* np.dot(X, pesos_entrada_oculta): Multiplicação de matriz das entradas com os pesos da camada oculta.
* np.dot(camada_oculta, pesos_oculta_saida): Multiplicação de matriz dos neurônios ocultos com os pesos da saída.
* As ativações são passadas pela função sigmoide.

## Exemplo de Forward Pass: 

* Suponha que X = [0, 1] e que os pesos iniciais sejam aleatórios.

```python
np.dot([0, 1], [[-0.5, 0.2, 0.7], [0.4, -0.3, 0.8]])
# Resultado: [0.4, -0.3, 0.8]  # Sinais da camada oculta

sigmoid([0.4, -0.3, 0.8]) 
# Resultado: [0.60, 0.42, 0.69]  # Ativação dos neurônios ocultos

np.dot([0.60, 0.42, 0.69], [[-0.3], [0.5], [-0.6]])
# Resultado: [-0.32]  # Sinal da camada de saída

sigmoid(-0.32) 
# Resultado: 0.42  # Saída final (ainda com erro)
```

## Cálculo do Erro

```python
erro = y - camada_saida
```
* O erro é a diferença entre a saída esperada (y) e a saída calculada.


## Backward Pass (Backpropagation)

```python
d_saida = erro * sigmoid_derivative(camada_saida)  # Gradiente da saída
erro_oculta = d_saida.dot(pesos_oculta_saida.T)  # Propagação do erro para a camada oculta
d_oculta = erro_oculta * sigmoid_derivative(camada_oculta)  # Gradiente da camada oculta
```
* Gradiente da saída: Calculamos quanto a saída precisa mudar.
* Erro da camada oculta: Multiplicamos o erro da saída pelos pesos da camada oculta.
* Gradiente da camada oculta: Calculamos quanto os neurônios ocultos devem ser ajustados.

## Atualização dos Pesos

```python
pesos_oculta_saida += camada_oculta.T.dot(d_saida) * learning_rate
pesos_entrada_oculta += X.T.dot(d_oculta) * learning_rate
```

* Os pesos são atualizados com base nos gradientes calculados.
* Multiplicação matricial garante que todas as conexões sejam ajustadas corretamente.

## Monitoramento do Erro

```python
if epoch % 1000 == 0:
    print(f"Erro na época {epoch}: {np.mean(np.abs(erro)):.4f}")
```

* A cada 1000 épocas, o erro médio absoluto é impresso.
* Isso permite verificar se o modelo está aprendendo corretamente.

## Saída

![image](https://github.com/user-attachments/assets/90bfff9e-2071-4242-a268-4c8aaf836074)

## Usando a função de ativação Tanh

```python
# Função de ativação tanh e sua derivada
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2
```

## Forward Pass 

```python
camada_oculta = tanh(np.dot(X, pesos_entrada_oculta))  # Entrada -> Oculta
camada_saida = tanh(np.dot(camada_oculta, pesos_oculta_saida))  # Oculta -> Saída
```

## Backpropagation

```python
d_saida = erro * tanh_derivative(camada_saida)  # Gradiente da saída
erro_oculta = d_saida.dot(pesos_oculta_saida.T)  # Propagação do erro para a camada oculta
d_oculta = erro_oculta * tanh_derivative(camada_oculta)  # Gradiente da oculta
```

## Saída 

![image](https://github.com/user-attachments/assets/256cc567-6fc4-4479-af80-b16e1e52014e)


## Usando tensorflow / keras

* <strong> Bibliotecas </strong>
<br>

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

## Definição dos dados de entrada (X) e saída esperada (y)

```python
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR
```

## Construção do modelo com Keras

```python
model = Sequential([
    Dense(4, input_dim=2, activation='tanh'),  # Camada oculta com 4 neurônios
    Dense(1, activation='sigmoid')  # Camada de saída com 1 neurônio
])
```

## Compilação do modelo

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## Treinamento do modelo

```python
model.fit(X, y, epochs=1000, verbose=0)  # Treina por 1000 épocas
```

## Testando o modelo treinado

```python
print("\nSaídas previstas pelo modelo após treinamento:")
predictions = model.predict(X)
print(np.round(predictions))  # Arredondamos para 0 ou 1
```

## Saída 

![image](https://github.com/user-attachments/assets/867befcf-edb9-4140-8371-4e7c4dfc5e2e)


## Referências 

<strong>Livro: Redes Neurais Artificiais Para Engenharia e Ciências Aplicadas. Fundamentos Teóricos e Aspectos Práticos </strong>

