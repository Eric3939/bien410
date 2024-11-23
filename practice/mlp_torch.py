import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init




# loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(output, label)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for _ in range(iters):
#     y_pred = model(X)
#     loss = loss_fn(y_pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    





if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score


    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=(0.6, 0.4), random_state=42)
    y = y.reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # pytorch
    # check GPU
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No GPU detected, using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()    
    )
    model = model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        y_pred = model(torch.from_numpy(X_train).float().to(device))
        loss = loss_fn(y_pred, torch.from_numpy(y_train).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}, Loss: {loss.item():.4f}]")

    y_pred = model(torch.from_numpy(X_test).float().to(device)).detach().cpu().numpy()
    y_pred = (y_pred > 0.5).astype(int)



    # # sklearn MLP
    # mlp = MLPClassifier(hidden_layer_sizes=(15, 10, 5), max_iter=1000)
    # mlp.fit(X_train, y_train)
    # y_pred = mlp.predict(X_test)



    # # hand-coded MLP
    # mlp = MLP(20, [15, 10, 5], 1)
    # mlp.fit(X_train, y_train, epochs=6000, rate=0.1)
    # mlp.save_parameters('param.txt')
    # mlp.load_parameters('param.txt')
    # y_pred = mlp.predict(X_test)

    # evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")