from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import BlipProcessor, BertTokenizer, BlipImageProcessor, AutoProcessor, BlipConfig, BlipForQuestionAnswering
from transformers import GenerationConfig
import numpy as np

image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
text_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
config = BlipConfig.from_pretrained("Salesforce/blip-vqa-base")
# generation_config = GenerationConfig.from_pretrained("Salesforce/blip-vqa-base")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
tokenizer = BertTokenizer.from_pretrained("Salesforce/blip-vqa-base")
# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
model2 = torch.load("model/model.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image):
    # Resize the image to the expected size
    resized_image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(resized_image)
    # Normalize the image
    normalized_image = image_array / 255.0
    # Convert the image to a tensor
    image_tensor = torch.tensor(normalized_image).permute(2, 0, 1).float()
    return image_tensor


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['imageFile']
    image_filename = image_file.filename
    image_path = "./images/" + image_filename
    image_file.save(image_path) 
    question = request.form['question']



    # Process the image
    image = Image.open(image_path)
    image_encoding = image_processor(image, do_resize=True, size=(128, 128), return_tensors="pt")

    # Process the question
    encoding = text_processor(None, question, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    for k, v in encoding.items():
        encoding[k] = v.squeeze()

    inputs = processor(image, question, return_tensors="pt")
    decoder_input_ids = tokenizer(question, return_tensors="pt").input_ids
    # Prepare the input for the model
    input_data = {
        "pixel_values": image_encoding["pixel_values"],
        "input_ids": encoding["input_ids"],
        **inputs,
        # "decoder_input_ids": decoder_input_ids
    }

    # Make the prediction
    outputs = model2.generate(**input_data)
    print(outputs)
    predicted_answer = text_processor.decode(outputs[0], skip_special_tokens=True)
    # predicted_answer = text_processor.decode(outputs[0], skip_special_tokens=True)
    print(predicted_answer)

    # Return the predicted answer to the user interface
    return render_template('index.html')
    



if __name__ == '__main__':
    app.run(port=4000, debug=True)



   # Process the image
    # # Load the image using PIL
    # image = Image.open(image_path).convert('RGB')

    # inputs = processor(image, question, return_tensors="pt")
    # pixel_values = inputs['pixel_values']
    # input_ids = inputs['input_ids']
    # decoder_input_ids = tokenizer(question, return_tensors="pt").input_ids

    # # Forward pass
    # outputs = model.generate(**inputs)
    # predicted_answer = processor.decode(outputs[0], skip_special_tokens=True)
    # image = Image.open(image_path)
    # preprocessed_image = preprocess_image(image)
    # image_tensor = torch.tensor(preprocessed_image).unsqueeze(0).to(device)
    # inputs = processor(image, question, return_tensors="pt")
    # input_data = {
    # 'pixel_values': preprocessed_image.unsqueeze(0).to(device),  # Assuming 'device' is the target device
    # 'input_ids': inputs['input_ids'].to(device)
    # }
    # outputs = model.generate(**input_data)
    # predicted_answer = processor.decode(outputs[0], skip_special_tokens=True)
    # print(predicted_answer)