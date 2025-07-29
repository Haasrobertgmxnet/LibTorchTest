#pragma once

#include <torch/torch.h>
#include "Global.h"

namespace Beam {
    // Physics informed neural network
    class PINN : public torch::nn::Module {
    public:
        PINN() {
            // Network definition: 1 Input x -> 5 Hidden -> 5 Hidden -> 1 Output (u)
            fc1 = register_module("fc1", torch::nn::Linear(1, 5));
            fc2 = register_module("fc2", torch::nn::Linear(5, 5));
            fc3 = register_module("fc3", torch::nn::Linear(5, 1));

            // Gewichte initialisieren
            torch::nn::init::xavier_uniform_(fc1->weight);
            torch::nn::init::xavier_uniform_(fc2->weight);
            torch::nn::init::xavier_uniform_(fc3->weight);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::tanh(fc1->forward(x));
            x = torch::tanh(fc2->forward(x));
            x = fc3->forward(x);
            return x;
        }

    private:
        torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
    };

    // Losses
    struct Losses {
        torch::Tensor total;
        torch::Tensor physics;
        torch::Tensor boundary;
    };

    constexpr auto adam_epochs = uint16_t{ 500 };
    constexpr auto adam_epochs_diff = uint16_t{ 100 };

    // Trainingsdaten generieren
    std::pair<torch::Tensor, torch::Tensor> generate_training_data_(int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        // Zufaellige Punkte im Raum-Zeit-Bereich
        auto x = torch::rand({ n_points, 1 }, options) * 1.0;  // x in [0, 1]

        return std::make_pair(x, x);
    }

    std::pair<torch::Tensor, torch::Tensor> generate_boundary_conditions(int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        auto x_left = torch::full({ n_points, 1 }, 0.0, options);
        auto x_right = torch::full({ n_points, 1 }, 1.0, options);

        auto input = torch::cat({ x_left, x_right }, 0);
        auto target = torch::zeros({ 2 * n_points, 1 }, options);

        return { input, target };
    }


    std::pair<torch::Tensor, torch::Tensor> generate_training_data(int n_points = 100) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        // x in [0, 1]
        // auto x = torch::linspace(0.0, 1.0, n_points, options).unsqueeze(1);
        auto x = torch::rand({ n_points, 1 }, options) * 1.0;  // x in [0, 1]

        // hier evtl. auch Zielwerte (z.B. Startwerte, falls bekannt)
        auto y = torch::zeros_like(x); // dummy target, falls nötig

        return { x, y };
    }

    // Randbedingungen generieren
    torch::Tensor boundary_loss(PINN& model) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

        // Punkt x = 0
        auto x0 = torch::zeros({ 1, 1 }, options).set_requires_grad(true);
        auto u0 = model.forward(x0);
        auto du0 = torch::autograd::grad({ u0 }, { x0 }, /*grad_outputs=*/{ torch::ones_like(u0) },
            /*retain_graph=*/true, /*create_graph=*/true)[0];

        // Punkt x = 1
        auto x1 = torch::ones({ 1, 1 }, options).set_requires_grad(true);
        auto u1 = model.forward(x1);

        auto du1 = torch::autograd::grad({ u1 }, { x1 }, { torch::ones_like(u1) },
            /*retain_graph=*/true, /*create_graph=*/true)[0];
        auto d2u1 = torch::autograd::grad({ du1 }, { x1 }, { torch::ones_like(du1) },
            /*retain_graph=*/true, /*create_graph=*/true)[0];
        auto d3u1 = torch::autograd::grad({ d2u1 }, { x1 }, { torch::ones_like(d2u1) },
            /*retain_graph=*/true, /*create_graph=*/true)[0];

        // Randverlust
        auto loss = torch::mse_loss(u0, torch::zeros_like(u0)) +
            torch::mse_loss(du0, torch::zeros_like(du0)) +
            torch::mse_loss(d2u1, torch::zeros_like(d2u1)) +
            torch::mse_loss(d3u1, torch::zeros_like(d3u1));

        return loss;
    }

    torch::Tensor physics_loss(PINN& model, torch::Tensor input, float EI = 1.0f) {
        try {
            auto x = input.clone().requires_grad_(true);  // input in R^{N×1}, nur x

            auto u = model.forward(x);
            auto ones = torch::ones_like(u);

            // Erste Ableitung
            auto du_dx = torch::autograd::grad({ u }, { x }, { ones }, Global::keep_graph, true)[0];

            // Zweite Ableitung
            auto d2u_dx2 = torch::autograd::grad({ du_dx }, { x }, { ones }, Global::keep_graph, true)[0];

            // Dritte Ableitung
            auto d3u_dx3 = torch::autograd::grad({ d2u_dx2 }, { x }, { ones }, Global::keep_graph, true)[0];

            // Vierte Ableitung
            auto d4u_dx4 = torch::autograd::grad({ d3u_dx3 }, { x }, { ones }, Global::keep_graph, true)[0];

            // Rechte Seite der DGL: z. B. q(x) = konst. = 1
            auto q = torch::ones_like(x);  // oder eine Funktion von x

            // Euler-Bernoulli-Gleichung: E*I*d4u/dx4 = q
            auto residual = EI * d4u_dx4 - q;

            return torch::mean(residual.pow(2));
        }
        catch (const std::exception& e) {
            std::cerr << "Fehler in physics_loss: " << e.what() << std::endl;
            return torch::tensor(0.0f, torch::requires_grad(true));
        }
    }

    // L2 Regularisation
    torch::Tensor compute_l2_regularization(PINN& model, float lambda_reg) {
        if (lambda_reg <= 0.0f) {
            return torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
        }

        torch::Tensor l2 = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
        for (const auto& param : model.parameters()) {
            l2 += torch::norm(param, 2).pow(2);
        }

        return lambda_reg * l2;
    }

    Losses compute_losses(PINN& model, torch::Tensor physics_input) {
        try {
            // Physics Loss (PDE-Residuum)
            auto loss_physics = physics_loss(model, physics_input);

            // Boundary Condition Loss
            auto loss_boundary = boundary_loss(model);

            auto total = loss_physics + 2.0f * loss_boundary;
            return { total, loss_physics, loss_boundary };
        }
        catch (const std::exception& e) {
            std::cerr << "Fehler in compute_losses: " << e.what() << std::endl;
            auto dummy = torch::tensor(1.0f, torch::requires_grad(true));
            return { dummy, dummy, dummy };
        }
    }


    

    // Gesamte Loss-Funktion (mit Error-Handling)
    //torch::Tensor total_loss(PINN& model,
    //    torch::Tensor physics_input) {

    //    try {
    //        //// Physics Loss (PDE-Residuum)
    //        //auto loss_physics = physics_loss(model, physics_input);

    //        //// Boundary Condition Loss
    //        //auto loss_boundary = boundary_loss(model);

    //        //// Gewichtete Summe der Verluste
    //        //auto total = 1.0f * loss_physics + 2.0f * loss_boundary;

    //        return compute_losses(model, physics_input).total;

    //    }
    //    catch (const std::exception& e) {
    //        std::cout << "Fehler in total_loss: " << e.what() << std::endl;
    //        return torch::tensor(1.0f, torch::requires_grad(true));
    //    }
    //}

}
