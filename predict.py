import torch
import timm
from PIL import Image
from torchvision import transforms

MODEL_PATH = "final_fake_real_swin_model.pth"
IMAGE_PATH = "Dentist.png"

device = torch.device("cpu")

model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=1
)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probability = torch.sigmoid(output).item()

label = "fake" if probability >= 0.5 else "real"

print("Prediction:", label)
print("Probability:", probability)