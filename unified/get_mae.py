import pickle

for path, label in [
    ("shelter_model_output_v0/shelter_model_artifacts.pkl", "v0"),
    ("shelter_model_output/shelter_model_artifacts.pkl",    "latest"),
]:
    with open(path, "rb") as f:
        art = pickle.load(f)
    print(f"Stored MAE ({label}): {art['train_mae_days']:.1f} days")
