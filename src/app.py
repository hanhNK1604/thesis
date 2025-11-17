import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.fm_latent_module import FlowMatchingLatentModule
import streamlit as st 
from PIL import Image
import torch 
from torchvision import transforms 

# checkpoint = '/workspace/thesis/checkpoints/fm_latent/isic/epoch=149-step=125400.ckpt'
# module = FlowMatchingLatentModule.load_from_checkpoint(
#     checkpoint_path = checkpoint, 
#     vae_image_path = "/workspace/thesis/checkpoints/vae_image/isic/epoch=249-step=209000.ckpt",
#     vae_mask_path = "/workspace/thesis/checkpoints/vae_mask/isic/epoch=99-step=83600.ckpt" 
# ) 

checkpoint = '/workspace/thesis/checkpoints/fm_latent/clinic/epoch=439-step=54121.ckpt'
module = FlowMatchingLatentModule.load_from_checkpoint(
    checkpoint_path = checkpoint, 
    vae_image_path = "/workspace/thesis/checkpoints/vae_image/clinic/epoch=249-step=30750.ckpt",
    vae_mask_path = "/workspace/thesis/checkpoints/vae_mask/clinic/epoch=199-step=24600.ckpt" 
) 

module.eval().freeze() 

def response_result(image): 
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((256, 256)), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = transform(image).to("cuda")
    image = torch.unsqueeze(image, dim=0) 
    z_image, _ = module.vae_image_module.vae_model.encode(image)
    mask_pred = module.sample_n(z_image=z_image, n=5) 
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