import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def update_and_predict():
    print("Updating data and making predictions...")

    # Step 1: Download Nifty 50 Data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    nifty_ticker = "^NSEI"
    nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date, interval="1d")

    if nifty_data.empty:
        raise ValueError("No data retrieved for Nifty 50. Check ticker or date range.")

    if isinstance(nifty_data.columns, pd.MultiIndex):
        nifty_data.columns = nifty_data.columns.get_level_values(0)

    nifty_data = nifty_data[["Open", "High", "Low", "Close"]]
    nifty_data["Prev Close"] = nifty_data["Close"].shift(1)
    nifty_data.dropna(subset=["Prev Close"], inplace=True)

    nifty_data["% High Move"] = ((nifty_data["High"] - nifty_data["Open"]) / nifty_data["Open"]) * 100
    nifty_data["% Low Move"] = ((nifty_data["Low"] - nifty_data["Open"]) / nifty_data["Open"]) * 100
    nifty_data["% Close Move"] = ((nifty_data["Close"] - nifty_data["Prev Close"]) / nifty_data["Prev Close"]) * 100

    features = nifty_data[["Prev Close", "Open", "High", "Low"]]
    target = np.where(nifty_data["% Close Move"] > 0, 1, 0)

    if len(np.unique(target)) > 1:
        stratify_option = target
    else:
        stratify_option = None

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=stratify_option)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel="linear", C=1.0)
    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    avg_high_move = nifty_data["% High Move"].mean()
    avg_low_move = nifty_data["% Low Move"].mean()
    avg_close_move = nifty_data["% Close Move"].mean()

    latest_features = scaler.transform([nifty_data.iloc[-1][["Prev Close", "Open", "High", "Low"]].values])
    tomorrow_prediction = svm_model.predict(latest_features)[0]

    latest_close = nifty_data.iloc[-1]["Close"]
    tomorrow_target_high = latest_close * (1 + avg_high_move / 100)
    tomorrow_target_low = latest_close * (1 + avg_low_move / 100)
    tomorrow_buy_price = latest_close * (1 - avg_low_move / 200)

    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Tomorrow's Market Prediction: {'Up' if tomorrow_prediction == 1 else 'Down'}")
    print(f"Tomorrow's Target High: {tomorrow_target_high:.2f}")
    print(f"Tomorrow's Target Low: {tomorrow_target_low:.2f}")
    print(f"Suggested Buying Price: {tomorrow_buy_price:.2f}")

    # --- MATPLOTLIB LOVE - PRICE PLOT ---
    plt.figure(figsize=(14, 6))
    plt.plot(nifty_data.index[-60:], nifty_data["Close"].iloc[-60:], label="Close Price", color='blue')
    plt.axhline(y=latest_close, color='gray', linestyle='--', label="Latest Close")
    plt.axhline(y=tomorrow_target_high, color='green', linestyle='--', label="Predicted High")
    plt.axhline(y=tomorrow_target_low, color='red', linestyle='--', label="Predicted Low")
    plt.axhline(y=tomorrow_buy_price, color='purple', linestyle=':', label="Buy Price")
    plt.title("Nifty 50 - Last 60 Days with SVM Prediction Levels")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- MATPLOTLIB LOVE - PCA CLASSIFICATION VISUALIZATION ---
    # PCA for 2D projection of features
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(scaler.transform(features))
    y_full_pred = svm_model.predict(scaler.transform(features))

    # Plot true labels
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=target, cmap='coolwarm', edgecolor='k', s=40, alpha=0.7)
    plt.title("SVM Classification - PCA 2D Projection (True Labels)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.colorbar(scatter, label="Class (0 = Down, 1 = Up)")
    plt.tight_layout()
    plt.show()

    # Plot predicted labels
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_full_pred, cmap='coolwarm', edgecolor='k', s=40, alpha=0.7)
    plt.title("SVM Classification - PCA 2D Projection (Predicted Labels)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.colorbar(scatter, label="Prediction (0 = Down, 1 = Up)")
    plt.tight_layout()
    plt.show()

# Run it
update_and_predict()