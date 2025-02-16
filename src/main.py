import torch
from model import DeepSeekMoE
from multi_token_prediction import MultiTokenPrediction


def main():
    dim = 512
    vocab_size = 32000
    num_experts = 16
    top_k = 4

    # Modelo principal
    moe_layer = DeepSeekMoE(num_experts, top_k, dim, 2048)
    mtp = MultiTokenPrediction(dim, depth=1, vocab_size=vocab_size)

    # Forward pass simulado
    x = torch.randn(2, 10, dim)  # Batch 2, seq 10, dim 512

    # Passagem pelo MoE
    moe_output = moe_layer(x)

    # Previsão multi-token (depth=1)
    next_token_emb = torch.randn(2, 10, dim)  # Embedding do próximo token
    prediction = mtp(moe_output, next_token_emb)

    print("Saída do MoE:", moe_output.shape)
    print("Previsão multi-token:", prediction.shape)

    # Atualização de balanceamento (simulando passo de treino)
    moe_layer.update_balance()


if __name__ == "__main__":
    main()
