import torch

path = "final_fake_real_swin_model.pth"

print("Loading model file...")
data = torch.load(path, map_location="cpu")

print("\nLoaded successfully")
print("Type:", type(data))

if isinstance(data, dict):
    print("\nTop level keys:")
    for key in data.keys():
        print(key)

    print("\nChecking first few items:")
    for i, key in enumerate(data.keys()):
        print(f"{key}: {type(data[key])}")
        if i >= 10:
            break
else:
    print("\nThis file is not a dictionary checkpoint.")
    print(data)