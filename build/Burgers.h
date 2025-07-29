#pragma once

#include <torch/torch.h>
#include "Global.h"

namespace Burgers {
    // Physics informed neural network
    class PINN : public torch::nn::Module {
    public:
        PINN() {
            // Netzwerk-Definition: 2 Inputs (x, t) -> 50 Hidden -> 1 Output (u)
            fc1 = register_module("fc1", torch::nn::Linear(2, 50));
            fc2 = register_module("fc2", torch::nn::Linear(50, 80));
            fc3 = register_module("fc3", torch::nn::Linear(80, 50));
            fc4 = register_module("fc4", torch::nn::Linear(50, 1));

            // Gewichte initialisieren
            torch::nn::init::xavier_uniform_(fc1->weight);
            torch::nn::init::xavier_uniform_(fc2->weight);
            torch::nn::init::xavier_uniform_(fc3->weight);
            torch::nn::init::xavier_uniform_(fc4->weight);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::tanh(fc1->forward(x));
            x = torch::tanh(fc2->forward(x));
            x = torch::tanh(fc3->forward(x));
            x = fc4->forward(x);
            return x;
        }

    private:
        torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr };
    };

    // Losses
    struct Losses {
        torch::Tensor total;
        torch::Tensor physics;
        torch::Tensor init;
        torch::Tensor boundary;
    };

    //constexpr auto adam_epochs = uint16_t{ 500 };
    //constexpr auto adam_epochs_diff = uint16_t{ 100 };

    constexpr auto adam_epochs = uint16_t{ 10 };
    constexpr auto adam_epochs_diff = uint16_t{ 2 };

    // Trainingsdaten generieren
    std::pair<torch::Tensor, torch::Tensor> generate_training_data(const int n_points = 1000) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        // Zufällige Punkte im Raum-Zeit-Bereich
        auto x = torch::rand({ n_points, 1 }, options) * 2.0 - 1.0;  // x in [-1, 1]
        auto t = torch::rand({ n_points, 1 }, options) * 1.0;        // t in [0, 1]

        auto input = torch::cat({ x, t }, 1);
        return std::make_pair(input, torch::cat({ x, t }, 1));
    }

    // Anfangsbedingungen generieren
    std::pair<torch::Tensor, torch::Tensor> generate_initial_conditions(const int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        auto x = torch::linspace(-1.0, 1.0, n_points, options).unsqueeze(1);
        auto t = torch::zeros({ n_points, 1 }, options);

        // Anfangsbedingung: u(x, 0) = -sin(pi*x)
        auto u_init = -torch::sin(M_PI * x);

        auto input = torch::cat({ x, t }, 1);
        return std::make_pair(input, u_init);
    }

    // Randbedingungen generieren
    std::pair<torch::Tensor, torch::Tensor> generate_boundary_conditions(const int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        auto t = torch::linspace(0.0, 1.0, n_points, options).unsqueeze(1);

        // Linke Grenze: x = -1
        auto x_left = torch::full({ n_points, 1 }, -1.0, options);
        auto input_left = torch::cat({ x_left, t }, 1);

        // Rechte Grenze: x = 1
        auto x_right = torch::full({ n_points, 1 }, 1.0, options);
        auto input_right = torch::cat({ x_right, t }, 1);

        // Homogene Dirichlet-Randbedingungen: u(-1, t) = u(1, t) = 0
        auto u_boundary = torch::zeros({ n_points, 1 }, options);

        auto input = torch::cat({ input_left, input_right }, 0);
        auto target = torch::cat({ u_boundary, u_boundary }, 0);

        return std::make_pair(input, target);
    }

    // Physics-basierte Loss-Funktion 
    torch::Tensor physics_loss(PINN& model, const torch::Tensor& input) {
        try {
            auto x = input.slice(1, 0, 1).clone().requires_grad_(true);
            auto t = input.slice(1, 1, 2).clone().requires_grad_(true);
            auto input_grad = torch::cat({ x, t }, 1);

            auto u = model.forward(input_grad);

            auto ones = torch::ones_like(u);

            // Gradienten aller Eingaben (x und t) auf einmal
            auto grads = torch::autograd::grad(
                { u }, { input_grad },
                { ones },
                /*retain_graph=*/Global::keep_graph,
                /*create_graph=*/true
            )[0];

            auto u_x = grads.slice(1, 0, 1);
            auto u_t = grads.slice(1, 1, 2);

            auto residual = u_t + u * u_x;

            return torch::mean(residual.pow(2));
        }
        catch (const std::exception& e) {
            std::cerr << "Fehler in physics_loss: " << e.what() << std::endl;
            return torch::tensor(0.0f, torch::requires_grad(true));
        }
    }

    Losses compute_losses(PINN& model,
        const torch::Tensor& physics_input,
        const torch::Tensor& init_input, const torch::Tensor& init_target,
        const torch::Tensor& boundary_input, const torch::Tensor& boundary_target) {
        try {
            auto loss_physics = physics_loss(model, physics_input);
            auto u_init_pred = model.forward(init_input);
            auto loss_init = torch::mse_loss(u_init_pred, init_target);
            auto u_boundary_pred = model.forward(boundary_input);
            auto loss_boundary = torch::mse_loss(u_boundary_pred, boundary_target);

            auto total = loss_physics + 2.0f * loss_init + 2.0f * loss_boundary;

            return { total, loss_physics, loss_init, loss_boundary };
        }
        catch (const std::exception& e) {
            std::cerr << "Fehler in compute_losses: " << e.what() << std::endl;
            auto dummy = torch::tensor(1.0f, torch::requires_grad(true));
            return { dummy, dummy, dummy, dummy };
        }
    }

    // Gesamte Loss-Funktion (mit Error-Handling)
    torch::Tensor total_loss(PINN& model,
        const torch::Tensor& physics_input,
        const torch::Tensor& init_input, const torch::Tensor& init_target,
        const torch::Tensor& boundary_input, const torch::Tensor& boundary_target) {

        try {
            // Physics Loss (PDE-Residuum)
            auto loss_physics = physics_loss(model, physics_input);

            // Initial Condition Loss
            auto u_init_pred = model.forward(init_input);
            auto loss_init = torch::mse_loss(u_init_pred, init_target);

            // Boundary Condition Loss
            auto u_boundary_pred = model.forward(boundary_input);
            auto loss_boundary = torch::mse_loss(u_boundary_pred, boundary_target);

            // Gewichtete Summe der Verluste
            // auto total = loss_physics + 10.0f * loss_init + 10.0f * loss_boundary;
            auto total = 1.0f * loss_physics + 1.0f * loss_init + 1.0f * loss_boundary;

            return total;

        }
        catch (const std::exception& e) {
            std::cout << "Fehler in total_loss: " << e.what() << std::endl;
            return torch::tensor(1.0f, torch::requires_grad(true));
        }
    }
}
