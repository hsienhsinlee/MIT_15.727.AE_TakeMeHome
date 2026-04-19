import pickle

with open("shelter_model_output_cats/cat_model_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

print(f"Stored Cats MAE: {art['train_mae_days']:.1f} days")
