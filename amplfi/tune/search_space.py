from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
}
