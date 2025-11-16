from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app = Flask(__name__)

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None

    if request.method == "POST":
        image_file = request.files["image"]

        if image_file:
            image = Image.open(image_file.stream)

            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

    return render_template("index.html", caption=caption)


if __name__ == "__main__":
    app.run(debug=True)
