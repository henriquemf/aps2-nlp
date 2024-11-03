import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from embedding import data, model, EmbeddingAutoencoder, input

# Carregar o modelo autoencoder treinado e embeddings otimizados
autoencoder = EmbeddingAutoencoder(input_dim)  # Defina o input_dim conforme o arquivo original
autoencoder.load_state_dict(torch.load('embedding_autoencoder.pth'))
autoencoder.eval()

# Carregar embeddings originais e otimizados
embeddings = model.encode(data['text'].tolist(), convert_to_tensor=True)  # Se os embeddings não estiverem salvos
optimized_embeddings = autoencoder.encoder(embeddings).detach().numpy()

# Projeção com t-SNE dos embeddings originais (pré-treinados)
tsne_original = TSNE(n_components=2, random_state=42)
original_2d = tsne_original.fit_transform(embeddings.cpu().numpy())

# Projeção com t-SNE dos embeddings otimizados
tsne_optimized = TSNE(n_components=2, random_state=42)
optimized_2d = tsne_optimized.fit_transform(optimized_embeddings)

# Visualização dos embeddings originais
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(original_2d[:, 0], original_2d[:, 1], s=1, alpha=0.6)
plt.title("Embeddings Originais")

# Visualização dos embeddings otimizados
plt.subplot(1, 2, 2)
plt.scatter(optimized_2d[:, 0], optimized_2d[:, 1], s=1, alpha=0.6)
plt.title("Embeddings Otimizados")
plt.show()
