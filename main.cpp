#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "."
        << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

    // Einfacher Test
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << "Random Tensor:\n" << tensor << std::endl;

    // Test der automatischen Differentiation
    torch::Tensor x = torch::randn({ 1 }, torch::requires_grad(true));
    torch::Tensor y = x * x * 3;
    y.backward();

    std::cout << "x: " << x << std::endl;
    std::cout << "y = 3*x^2: " << y << std::endl;
    std::cout << "dy/dx = 6*x: " << x.grad() << std::endl;

    // Test ob CUDA verfügbar ist
    if (torch::cuda::is_available()) {
        std::cout << "CUDA ist verfuegbar!" << std::endl;
        std::cout << "CUDA Geraete: " << torch::cuda::device_count() << std::endl;
    }
    else {
        std::cout << "CUDA nicht verfuegbar - läuft auf CPU" << std::endl;
    }

    return 0;
}