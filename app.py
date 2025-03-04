import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cpu")  # Use GPU for faster generation


def generate_image(prompt):
    image = pipe(prompt).images[0]  # Generate the image
    return image


st.title("Funny Image Generator")
st.write("Generate a funny image with any quirky prompt!")


prompt = st.text_input(
    "Enter your image prompt",
    "baby deadpool"
)


if st.button("Generate Image"):
    if prompt:  # Check if the prompt is not empty
        with st.spinner('Generating image...'):
            image = generate_image(prompt)  # Generate image from the prompt
            st.image(image, caption="Generated Image", use_column_width=True)

            # Convert the generated image to BytesIO for download
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            st.download_button("Download Image", buffered.getvalue(), file_name="generated_image.png")
    else:
        st.warning("Please enter a prompt!")  # Show a warning if prompt is empty
