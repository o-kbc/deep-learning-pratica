import torch

if __name__ == "__main__":

    t1 = torch.tensor(1.)
    print(f"T1: {t1}")
    print(f"T1 shape: {t1.shape}")

    t2 = torch.tensor([1., 2.])
    print(f"T2: {t2}")
    print(f"T2 shape: {t2.shape}")

    t3 = torch.tensor([[1., 2.], [3., 4.]])
    print(f"T3: {t3}")
    print(f"T3 shape: {t3.shape}")

    print(f"T3 + T3: {(t3 + t3)}")
    print(f"Type: {(t3 + t3).type()}")
    print(f"Flattened: {(t3 + t3).flatten()}")
    print(f"Reshaped: {(t3 + t3).reshape(4,1)}")
    print(f"un-Squeezed: {(t3 + t3).unsqueeze(dim=0)}")

    t4 = torch.randn((2, 1, 5, 5))

    t5 = torch.randn((2, 1, 5, 5))

    print(f"T4: {t4}")
    print(f"T5: {t5}")

    print(f"DIFF: {t5 - t4}")

