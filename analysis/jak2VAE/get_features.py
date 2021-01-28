from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu

vae = VAEUtils(directory=".")

smiles1 = mu.canon_smiles('CSCC(=O)NNC(=O)c1c(C)oc(C)c1C')

X_1 = vae.smiles_to_hot(smiles1, canonize_smiles=True)
print(X_1)
z_1 = vae.encode(X_1)
print(z_1)
X_R = vae.decode(z_1)
print(X_R)
