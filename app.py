import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

st.set_page_config(
    page_title="Aritmética de características",
    page_icon="🧬",
    layout="centered"
)

st.title("Aritmética de características")
st.markdown("""
Use os sliders da sidebar para manipular o **Espaço Latente** e realizar aritmética de características.
""")

latent_dim = 100 

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

@st.cache_resource
def load_data():
    device = torch.device("cpu")
    
    netG = Generator(ngpu=0).to(device)
    
    if os.path.exists("netG_celeba_v1.pth"):
        state_dict = torch.load("netG_celeba_v1.pth", map_location=device)
        netG.load_state_dict(state_dict)
        netG.eval()
    else:
        st.error("Erro: Arquivo do modelo (netG_celeba_v1.pth) não encontrado.")
        return None, None

    vetores = {}
    try:
        vetores['sorriso'] = torch.load("vetor_sorriso.pth", map_location=device)
        vetores['loiro']   = torch.load("vetor_loiro.pth", map_location=device)
        vetores['genero']  = torch.load("vetor_genero.pth", map_location=device)
    except FileNotFoundError as e:
        st.warning(f"Aviso: Algum vetor não foi encontrado ({e}). Aritmética parcial.")
    
    return netG, vetores

netG, vetores = load_data()

if 'z_base' not in st.session_state:
    st.session_state['z_base'] = torch.randn(1, latent_dim, 1, 1)

if 'val_sorriso' not in st.session_state: st.session_state['val_sorriso'] = 0.0
if 'val_loiro' not in st.session_state: st.session_state['val_loiro'] = 0.0
if 'val_genero' not in st.session_state: st.session_state['val_genero'] = 0.0

def gerar_novo_rosto():
    st.session_state['z_base'] = torch.randn(1, latent_dim, 1, 1)

def resetar_sliders():
    st.session_state['val_sorriso'] = 0.0
    st.session_state['val_loiro'] = 0.0
    st.session_state['val_genero'] = 0.0

st.sidebar.header("Sliders")

val_sorriso = st.sidebar.slider("Sério <-> Sorriso", -3.0, 3.0, key='val_sorriso')
val_loiro   = st.sidebar.slider("Moreno <-> Loiro", -3.0, 3.0, key='val_loiro')
val_genero  = st.sidebar.slider("Homem <-> Mulher", -3.0, 3.0, key='val_genero')

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    st.button("Novo Rosto", on_click=gerar_novo_rosto)
with col_btn2:
    st.button("Resetar Sliders", on_click=resetar_sliders)

if netG:
    z = st.session_state['z_base'].clone()

    if 'sorriso' in vetores: z += vetores['sorriso'] * val_sorriso
    if 'loiro' in vetores:   z += vetores['loiro'] * val_loiro
    if 'genero' in vetores:  z += vetores['genero'] * val_genero

    with torch.no_grad():
        fake = netG(z).detach().cpu()

    img = np.transpose(fake[0].numpy(), (1, 2, 0))
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="Imagem Gerada", use_container_width=True)
        
    st.markdown(f"""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        Sorriso: {val_sorriso:.1f} | Loiro: {val_loiro:.1f} | Gênero: {val_genero:.1f}
    </div>
    """, unsafe_allow_html=True)