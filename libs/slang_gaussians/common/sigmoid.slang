// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

// Common sigmoid operations library for Slang shaders

namespace sigmoid {

// Standard sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))
[Differentiable]
float standard(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Steep sigmoid function: op_sigmoid(x) = 1 / (1 + exp(-k * (x - x0)))
// Used to weight noise by opacity - high opacity gaussians get more perturbation
[Differentiable]
float steep(float x, float k = 100.0f, float x0 = 0.995f) {
    return 1.0f / (1.0f + exp(-k * (x - x0)));
}

} // namespace sigmoid
