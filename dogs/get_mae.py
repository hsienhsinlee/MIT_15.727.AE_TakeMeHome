import pickle

with open("shelter_model_output_dogs/dog_model_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

print(f"Stored Dogs MAE: {art['train_mae_days']:.1f} days")
