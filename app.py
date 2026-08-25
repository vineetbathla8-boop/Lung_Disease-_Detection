from flask import Flask,render_template,jsonify,request
import torch
from torchvision import transforms
from PIL import Image
from src.model import Net
from src.xray_detector_model import Net_1

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

def image_transformation(img):
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img).convert("RGB")
    img = test_transform(img)

    return img


# loading the trained model
model = Net()

model.load_state_dict(
    torch.load(
        "/Users/vinni/Desktop/ Lung Disease Detection/models/trained_model.pth",
        map_location="cpu"
    )
)
model.eval()



## loading the xray detector model 
model_1 = Net_1()
model_1.load_state_dict(
    torch.load(
        "/Users/vinni/Desktop/ Lung Disease Detection/models/xray_detector.pth",
        map_location="cpu"
    )
)
model_1.eval()





## Home Page
@app.route("/",methods = ["GET"])
def home_page():
    return render_template("index.html")






## checking wheather uploaded image is belong to other images or X-Ray images

@app.route("/checker",methods = ["POST"])
def checker():
    img = request.files["inp_file"]
    image = image_transformation(img)
    
        # Add batch dimension
    image = image.unsqueeze(0)
    
    with torch.no_grad():
    
        output = model_1(image)
    
        probabilities = torch.exp(output)
    
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
    class_names = ["Others", "  chest_xray"]
    
    prediction = class_names[predicted_class.item()]
    confidence = confidence.item() * 100
    
    data = {
        "prediction": prediction,
        "confidence": round(confidence, 2)
        }
    
    if data["prediction"] == "Others":

        return render_template(
        "index.html",
        name="Others"
        )

    else:

        with torch.no_grad():

            output = model(image)

            probabilities = torch.exp(output)

            confidence, predicted_class = torch.max(
            probabilities, dim=1
            )

        class_names = ["NORMAL", "PNEUMONIA"]

        prediction = class_names[predicted_class.item()]
        confidence = confidence.item() * 100

        data = {
        "prediction": prediction,
        "confidence": round(confidence, 2)
        }

        return render_template(
          "index.html",
            name="chest_xray",
            **data
        )


    
if __name__=="__main__":
    app.run(debug=True)