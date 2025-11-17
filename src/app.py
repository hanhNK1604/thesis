import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.fm_latent_module import FlowMatchingLatentModule
import streamlit as st 
from PIL import Image
import torch 
from torchvision import transforms 

vae_checkpoint = '/workspace/thesis/checkpoints/fm_latent/isic/epoch=149-step=125400.ckpt'
vae_module = FlowMatchingLatentModule.load_from_checkpoint(vae_checkpoint) 
vae_module.eval().freeze() 

def response_result(image): 
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((256, 256)), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = transform(image).to("cuda")
    image = torch.unsqueeze(image, dim=0) 
    z_image, _ = vae_module.vae_image_module.vae_model.encode(image)
    mask_pred = vae_module.sample_n(z_image=z_image, n=5) 
    mask_pred = mask_pred[0]
    output_image = transforms.ToPILImage()(mask_pred.cpu().float())
    return  output_image
     
st.title("Demo")

uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Input")

    if st.button("Predict"):
        with st.spinner("Processing..."):
            output_image = response_result(input_image)

        st.success("Success!")
        st.image(output_image, caption="Output")