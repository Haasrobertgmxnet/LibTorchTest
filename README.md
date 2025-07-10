# LibTorchTest – Physics-Informed Neural Network (PINN) for Burgers' Equation

**Language / Sprache / Idioma:**  
[🇬🇧 English](#english) | [🇩🇪 Deutsch](#deutsch) | [🇪🇸 Español](#espa%C3%B1ol)

---

## English

This repository demonstrates a Physics-Informed Neural Network (PINN) implemented with LibTorch (PyTorch C++ API) to solve the **inviscid Burgers' equation**:

```
∂u/∂t + u ∂u/∂x = 0
```

**Initial condition:**  
u(x, 0) = -sin(π x)  
**Boundary conditions:**  
u(-1, t) = u(1, t) = 0

### Features

- Written in pure C++ using LibTorch
- No external Python code required
- Uses automatic differentiation to enforce the PDE
- Optimizes via L-BFGS (quasi-Newton)

### How to Build

```bash
git clone https://github.com/Haasrobertgmxnet/LibTorchTest.git
cd LibTorchTest
mkdir build && cd build
cmake ..
make
./LibTorchTest
```

---

## Deutsch

Dieses Repository demonstriert ein Physics-Informed Neural Network (PINN) mit LibTorch (PyTorch C++ API) zur Lösung der **invisziden Burgers-Gleichung**:

```
∂u/∂t + u ∂u/∂x = 0
```

**Anfangsbedingung:**  
u(x, 0) = -sin(π x)  
**Randbedingungen:**  
u(-1, t) = u(1, t) = 0

### Merkmale

- Komplett in C++ geschrieben (LibTorch)
- Keine Python-Komponenten notwendig
- Nutzt automatische Differentiation zur PDE-Erfüllung
- Optimierung mittels L-BFGS (quasi-Newton-Verfahren)

### Kompilierung

```bash
git clone https://github.com/Haasrobertgmxnet/LibTorchTest.git
cd LibTorchTest
mkdir build && cd build
cmake ..
make
./LibTorchTest
```

---

## Español

Este repositorio muestra una red neuronal informada por la física (PINN) usando LibTorch (la API C++ de PyTorch) para resolver la **ecuación de Burgers sin viscosidad**:

```
∂u/∂t + u ∂u/∂x = 0
```

**Condición inicial:**  
u(x, 0) = -sin(π x)  
**Condiciones de frontera:**  
u(-1, t) = u(1, t) = 0

### Características

- Escrito completamente en C++ con LibTorch
- No se requiere código Python
- Usa diferenciación automática para cumplir con la PDE
- Optimiza usando L-BFGS (cuasi-Newton)

### Compilación

```bash
git clone https://github.com/Haasrobertgmxnet/LibTorchTest.git
cd LibTorchTest
mkdir build && cd build
cmake ..
make
./LibTorchTest
```