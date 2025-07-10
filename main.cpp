#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <array>

constexpr auto keep_graph = bool{ true };
constexpr auto adam_epochs = uint16_t{ 500 };
constexpr auto adam_epochs_diff = uint16_t{ 100 };

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

// Trainingsdaten generieren
std::pair<torch::Tensor, torch::Tensor> generate_training_data(int n_points = 1000) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

    // Zufällige Punkte im Raum-Zeit-Bereich
    auto x = torch::rand({ n_points, 1 }, options) * 2.0 - 1.0;  // x ∈ [-1, 1]
    auto t = torch::rand({ n_points, 1 }, options) * 1.0;        // t ∈ [0, 1]

    auto input = torch::cat({ x, t }, 1);
    return std::make_pair(input, torch::cat({ x, t }, 1));
}

// Anfangsbedingungen generieren
std::pair<torch::Tensor, torch::Tensor> generate_initial_conditions(int n_points = 100) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

    auto x = torch::linspace(-1.0, 1.0, n_points, options).unsqueeze(1);
    auto t = torch::zeros({ n_points, 1 }, options);

    // Anfangsbedingung: u(x, 0) = -sin(π*x)
    auto u_init = -torch::sin(M_PI * x);

    auto input = torch::cat({ x, t }, 1);
    return std::make_pair(input, u_init);
}

// Randbedingungen generieren
std::pair<torch::Tensor, torch::Tensor> generate_boundary_conditions(int n_points = 100) {
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
torch::Tensor physics_loss(PINN& model, torch::Tensor input) {
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
            /*retain_graph=*/keep_graph,
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
    torch::Tensor physics_input,
    torch::Tensor init_input, torch::Tensor init_target,
    torch::Tensor boundary_input, torch::Tensor boundary_target) {
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
    torch::Tensor physics_input,
    torch::Tensor init_input, torch::Tensor init_target,
    torch::Tensor boundary_input, torch::Tensor boundary_target) {

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

// Lösung visualisieren (einfache Konsolen-Ausgabe)
void visualize_solution(PINN& model, int grid_size = 20) {
    std::cout << "\nLösung der Burgers-Gleichung:\n";
    std::cout << "x\\t\t";

    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    // Zeitpunkte für Ausgabe
    std::vector<float> time_points = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

    for (auto t_val : time_points) {
        std::cout << "t=" << t_val << "\t";
    }
    std::cout << "\n";

    for (int i = 0; i < grid_size; i++) {
        float x_val = -1.0 + 2.0 * i / (grid_size - 1);
        std::cout << std::fixed << std::setprecision(2) << x_val << "\t";

        for (auto t_val : time_points) {
            auto x_tensor = torch::tensor({ x_val }, options).unsqueeze(0);
            auto t_tensor = torch::tensor({ t_val }, options).unsqueeze(0);
            auto input = torch::cat({ x_tensor, t_tensor }, 1);

            auto u_pred = model.forward(input);
            std::cout << std::setprecision(3) << u_pred.item<float>() << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "Physics-Informed Neural Network für Burgers-Gleichung\n";
    std::cout << "====================================================\n\n";

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA verfügbar - verwende GPU\n";
    }
    else {
        std::cout << "Verwende CPU\n";
    }

    auto model = std::make_shared<PINN>();
    model->to(device);

    // 

    auto [physics_input, _] = generate_training_data(2000);
    auto [init_input, init_target] = generate_initial_conditions(100);
    auto [boundary_input, boundary_target] = generate_boundary_conditions(100);

    physics_input = physics_input.to(device);
    init_input = init_input.to(device);
    init_target = init_target.to(device);
    boundary_input = boundary_input.to(device);
    boundary_target = boundary_target.to(device);

    float current_lr = 0.00001f;

    auto start_time = std::chrono::high_resolution_clock::now();

    // === PHASE 1: Adam Optimizer (grobes Training) ===

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(current_lr));

    

    std::array<Losses, adam_epochs / adam_epochs_diff> all_losses;
    
    std::cout << "\n[Phase 1] Adam Training...\n";

    for (int epoch = 0; epoch < adam_epochs; ++epoch) {
        try {
            optimizer.zero_grad();

            //auto loss = total_loss(*model, physics_input,
            //    init_input, init_target,
            //    boundary_input, boundary_target);

            auto losses = compute_losses(*model, physics_input,
                init_input, init_target,
                boundary_input, boundary_target);

            // NaN- oder Inf-Check
            if (torch::isnan(losses.total).any().item<bool>() || torch::isinf(losses.total).any().item<bool>()) {
                std::cerr << "Adam Warnung: Loss ist NaN oder Inf in Epoche " << epoch << std::endl;

                // Lernrate halbieren und im Optimizer einstellen
                current_lr *= 0.5f;
                std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
                for (auto& param_group : optimizer.param_groups()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
                    options.lr(current_lr);
                }
                continue;  // diese Epoche überspringen
            }

            losses.total.backward({}, /*keep_graph=*/keep_graph);

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            if (epoch % 200 == 0) {
                all_losses[epoch / adam_epochs_diff] = losses;
                std::cout << "Epoch " << epoch << "/" << adam_epochs
                    << " - Total: " << losses.total.item<float>()
                    << " | Physics: " << losses.physics.item<float>()
                    << " | Init: " << losses.init.item<float>()
                    << " | Boundary: " << losses.boundary.item<float>()
                    << " - LR: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Fehler in Adam Epoche " << epoch << ": " << e.what() << std::endl;
        }
    }

    //std::cout << "Losses:" << std::endl;
    //for (auto loss : all_losses) {
    //    std::cout << loss << std::endl;
    //}
    // === PHASE 2: LBFGS Optimizer (Feinabstimmung) ===
    std::cout << "\n[Phase 2] LBFGS Finetuning...\n";

    torch::optim::LBFGS lbfgs(model->parameters(),
        torch::optim::LBFGSOptions(1.0)
        .max_iter(20)
        .tolerance_grad(1e-7)
        .tolerance_change(1e-9)
        .history_size(100));

    int lbfgs_epochs = 0; // 200;

    for (int epoch = 0; epoch < lbfgs_epochs; ++epoch) {
        try {
            auto closure = [&]() -> torch::Tensor {
                lbfgs.zero_grad();

                auto loss = total_loss(*model, physics_input,
                    init_input, init_target,
                    boundary_input, boundary_target);

                // NaN- oder Inf-Check
                if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
                    std::cerr << "LBFGS Warnung: Loss ist NaN oder Inf in Epoche " << epoch << std::endl;

                    // Lernrate halbieren und im Optimizer einstellen
                    current_lr *= 0.5f;
                    std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
                    for (auto& param_group : lbfgs.param_groups()) {
                        auto& options = static_cast<torch::optim::LBFGSOptions&>(param_group.options());
                        options.lr(current_lr);
                    }
                    return torch::tensor(1.0f, torch::requires_grad(true));
                }


                loss.backward({}, /*keep_graph=*/keep_graph);
                return loss;
                };

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            torch::Tensor loss = lbfgs.step(closure);

            if (epoch % 10 == 0) {
                std::cout << "LBFGS Epoch " << epoch << "/" << lbfgs_epochs
                    << " - Loss: " << loss.item<float>() << " - Learning Rate: " << current_lr << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Fehler in LBFGS Epoche " << epoch << ": " << e.what() << std::endl;
        }
    }
    //for (int epoch = 0; epoch < num_epochs; epoch++) {
    //    try {
    //        optimizer.zero_grad();

    //        auto closure = [&]() -> torch::Tensor {
    //            optimizer.zero_grad();

    //            auto loss = total_loss(*model, physics_input,
    //                init_input, init_target,
    //                boundary_input, boundary_target);  // Dein Loss-Berechnungscode
    //            loss.backward({}, /*keep_graph=*/keep_graph);

    //            return loss;
    //            };

    //        

    //        auto loss = total_loss(*model, physics_input,
    //            init_input, init_target,
    //            boundary_input, boundary_target);

    //        // Debug-Ausgabe von Loss-Wert
    //        if (epoch % 100 == 0) {
    //            std::cout << "Epoch " << epoch << "/" << num_epochs
    //                << " - Loss: " << loss.item<float>()
    //                << " - Learning Rate: " << current_lr << std::endl;
    //        }

    //        // NaN- oder Inf-Check
    //        if (torch::isnan(loss).any().item<bool>() || torch::isinf(loss).any().item<bool>()) {
    //            std::cerr << "[WARNUNG] Loss ist NaN oder Inf in Epoche " << epoch << std::endl;

    //            // Lernrate halbieren und im Optimizer einstellen
    //            current_lr *= 0.5f;
    //            std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
    //            for (auto& param_group : optimizer.param_groups()) {
    //                auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
    //                options.lr(current_lr);
    //            }
    //            continue;  // diese Epoche überspringen
    //        }

    //        loss.backward({}, /*keep_graph=*/keep_graph);

    //        // Nach backward nochmal NaN prüfen in Gradients
    //        bool found_nan_grad = false;
    //        for (const auto& param : model->parameters()) {
    //            if (param.grad().defined() &&
    //                (torch::isnan(param.grad()).any().item<bool>() || torch::isinf(param.grad()).any().item<bool>())) {
    //                found_nan_grad = true;
    //                break;
    //            }
    //        }
    //        if (found_nan_grad) {
    //            std::cerr << "[WARNUNG] Gradient enthält NaN/Inf in Epoche " << epoch << std::endl;
    //            // Lernrate runtersetzen
    //            current_lr *= 0.5f;
    //            std::cerr << "Setze Lernrate auf " << current_lr << std::endl;
    //            for (auto& param_group : optimizer.param_groups()) {
    //                auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
    //                options.lr(current_lr);
    //            }
    //            optimizer.zero_grad();
    //            continue;
    //        }

    //        optimizer.step(closure);
    //        // optimizer.step();
    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "[FEHLER] Exception in Epoche " << epoch << ": " << e.what() << std::endl;
    //        std::cerr << "Abbruch des Trainings.\n";
    //        break;
    //    }
    //}

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nTraining abgeschlossen in " << duration.count() << " Sekunden\n";

    model->eval();
    visualize_solution(*model);

    return 0;
}
