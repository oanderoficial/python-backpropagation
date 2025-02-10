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

## Derivada da Função Sigmoid

```python
def sigmoid_derivative(x):
    return x * (1 - x)
```
* A derivada da função sigmoid é usada no backpropagation para calcular os ajustes necessários nos pesos.
* A fórmula vem da regra da derivação da função sigmoide.
