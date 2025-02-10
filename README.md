# Python-backpropagation

Exemplo básico de backpropagation em Python usando NumPy

* Usando backpropagation para treinar uma rede neural simples a resolver o problema XOR. A rede aprende ajustando os pesos automaticamente com base no erro.


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
pesos_entrada_oculta = np.random.uniform(-1, 1, (2, 3))  # 2 entradas -> 3 neurônios ocultos
pesos_oculta_saida = np.random.uniform(-1, 1, (3, 1))    # 3 neurônios ocultos -> 1 saída
```

* Os pesos das conexões são inicializados aleatoriamente no intervalo [-1, 1].
* pesos_entrada_oculta: Matriz (2x3) (2 neurônios de entrada → 3 neurônios na camada oculta)
* pesos_oculta_saida: Matriz (3x1) (3 neurônios da camada oculta → 1 neurônio na saída)

## Exemplo de pesos gerados:

```bash
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
learning_rate = 0.5
```
* Taxa de aprendizado controla o tamanho dos ajustes feitos nos pesos durante o treinamento.
* Valores muito grandes podem fazer o treinamento divergir, e valores muito pequenos podem torná-lo muito lento.


## 7) Loop de Treinamento

```python
for epoch in range(10000):
```
* O modelo será treinado por 10.000 iterações.

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
