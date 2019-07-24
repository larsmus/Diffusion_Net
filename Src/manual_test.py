import numpy as np
from Src.encoder_decoder import Encoder, Decoder
from Src.plotting import plot_reconstructed_digits

experiment = "1563885350"

# load and prepare data
embed = np.load(f"../Data/experiment_{experiment}/embed_train.npy")
x_train = np.load(f"../Data/experiment_{experiment}/x_train.npy")
y_train = np.load(f"../Data/experiment_{experiment}/y_train.npy")
eig = np.load(f"../Data/experiment_{experiment}/eig.npy", allow_pickle=True)

embedding_size = 32
input_size = 784
train_embed = embed[:, :embedding_size]

# train the encoder
encoded_subsample = Encoder(reg=0., embedding_size=embedding_size)
encoded_subsample.compile(loss="mean_squared_error", optimizer="adam")
encoded_subsample.train(x_train, train_embed, batch_size=128, epochs=15)

print("------------------------")
print("Done encoding")
print("------------------------")

predicted_embed_train = encoded_subsample.model.predict(x_train)

# train encoder
decoded_subsample = Decoder(reg=0.)
decoded_subsample.compile(loss="mean_squared_error", optimizer="adam")
decoded_subsample.train(predicted_embed_train, x_train, batch_size=128, epochs=15)

print("------------------------")
print("Done decoding")
print("------------------------")

reconstruct_subsample = decoded_subsample.model.predict(predicted_embed_train)

plot_reconstructed_digits(x_train, reconstruct_subsample)