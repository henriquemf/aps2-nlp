import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Carregar o dataset
data = pd.read_csv("art-db.csv")

# Concatenar campos textuais
data['text'] = data['title'].fillna('') + " " + data['artist_title'].fillna('') + " " + data['description'].fillna('')

# Carregar modelo sBERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # Versão rápida de sBERT

# Gerar embeddings
embeddings = model.encode(data['text'].tolist(), convert_to_tensor=True)

# Separar dados para treino e validação
train_embeddings, val_embeddings = train_test_split(embeddings, test_size=0.2, random_state=42)

# Configurar DataLoader para Pytorch
train_data = TensorDataset(train_embeddings)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Autoencoder para ajuste dos embeddings
class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Inicializar modelo, função de perda e otimizador
input_dim = embeddings.shape[1]
autoencoder = EmbeddingAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# Treinamento do autoencoder
epochs = 10
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[0]
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Salvar o modelo ajustado e embeddings otimizados
torch.save(autoencoder.state_dict(), 'embedding_autoencoder.pth')
optimized_embeddings = autoencoder.encoder(embeddings).detach().numpy()

# Projeção com t-SNE dos embeddings originais (pré-treinados)
tsne_original = TSNE(n_components=2, random_state=42)
original_2d = tsne_original.fit_transform(embeddings.cpu().numpy())

# Projeção com t-SNE dos embeddings otimizados
tsne_optimized = TSNE(n_components=2, random_state=42)
optimized_2d = tsne_optimized.fit_transform(optimized_embeddings)

# Visualização e salvamento dos embeddings originais
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(original_2d[:, 0], original_2d[:, 1], s=1, alpha=0.6)
plt.title("Embeddings Originais")
plt.savefig("embeddings_originais.png")  # Salvar imagem dos embeddings originais

# Visualização e salvamento dos embeddings otimizados
plt.subplot(1, 2, 2)
plt.scatter(optimized_2d[:, 0], optimized_2d[:, 1], s=1, alpha=0.6)
plt.title("Embeddings Otimizados")
plt.savefig("embeddings_otimizados.png")  # Salvar imagem dos embeddings otimizados

plt.show()
