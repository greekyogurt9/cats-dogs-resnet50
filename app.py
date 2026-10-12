import gradio as gr
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("cats_dogs_resnet50_compat.keras")


def predict_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)

    probability = model.predict(image, verbose=0)[0][0]

    if probability >= 0.5:
        return {
            "Dog": float(probability),
            "Cat": float(1 - probability)
        }
    else:
        return {
            "Cat": float(1 - probability),
            "Dog": float(probability)
        }


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2),
    title="🐱 Cat vs Dog 🐶",
    description="Upload an image and the model will classify it."
)

demo.launch(server_name="0.0.0.0", server_port=7860)