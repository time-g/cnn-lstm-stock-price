#!/usr/bin/env python3


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandas_ta  # isort:skip

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)
# plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (24, 8)
plt.rcParams["figure.dpi"] = 200
# plt.rcParams["axes.grid"] = False


def rolling_window(
    data: np.array, window_size=3, target_col=None, lstm_reshape=True
) -> (np.array, np.array):
    """
    data: numpy array.
    target_col: int, the target or y column.
    lstm_reshape: True, suitable shape for lstm model. pass False to reshape X to matrix-like shape.
    """
    x = []
    y = []
    for i in range(window_size, len(data)):
        x.append(data[i - window_size : i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    if target_col != None:
        y = y[:, target_col]
    if not lstm_reshape:
        # reshape x from (n_rows, n_step, n_features) to (n_rows, n_cols)
        n_rows, n_steps, n_features = x.shape
        n_cols = n_steps * n_features
        x = x.reshape(n_rows, n_cols)

    return x, y


def indicator_append(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = ["Open", "High", "Low", "Close"]
    df = df[ohlc]
    df.ta.rsi(append=True)
    df.ta.atr(append=True)
    df.ta.cci(length=20, append=True)

    # df[["MACD_12_26_9", "MACDs_12_26_9"]] = df.ta.macd()[
    #     ["MACD_12_26_9", "MACDs_12_26_9"]
    # ]
    df["MACDs_12_26_9"] = df.ta.macd()["MACDs_12_26_9"]

    df.dropna(inplace=True)
    return df


def save_model(model, fname: str = "tmp", ticker: str = "") -> str:
    folder = "models/"
    file = os.path.basename(fname)
    file = os.path.splitext(file)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    model_name = f"{folder}{file}{ticker}.keras"
    model.save(model_name)
    return model_name


def save_loss(loss: dict, fname: str = "tmp", ticker=""):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-loss{ticker}"
    pd.DataFrame(loss).to_csv(f"{fname}.csv")
    pd.DataFrame(loss).to_json(f"{fname}.json")


def plot_train_test_prediction(
    train,
    test,
    prediction,
    date,
    window_size,
    fname="tmp.png",
    ticker="",
    title="",
    show=True,
):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-plot-train-test{ticker}"
    train_size = len(train)
    shift_window_size = train_size + window_size

    # plt.figure(figsize=(16, 6), dpi=200)
    # shift the date to the right
    plt.plot(date[window_size:shift_window_size], train, lw=0.6)

    plt.plot(date[shift_window_size:], test, lw=0.6)
    plt.plot(date[shift_window_size:], prediction, lw=0.7)

    plt.legend(["train", "test", "prediction"])
    plt.title(title)
    plt.savefig(f"{fname}.png", dpi=200)
    if show:
        plt.show()


def plot_price_test_prediction(
    price,
    test,
    prediction,
    df_index,
    window_size,
    fname="tmp.png",
    ticker="",
    title="",
    show=True,
):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-plot-price-test{ticker}"
    train_size = len(price) - len(test) - window_size

    plt.plot(df_index, price, lw=0.6, label="real price")

    plt.plot(
        df_index[train_size + window_size :], test, lw=0.6, label="real(test) price"
    )
    plt.plot(
        df_index[train_size + window_size :],
        prediction,
        lw=0.8,
        label="predicted price",
        color="green",
    )
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.legend()

    plt.savefig(f"{fname}.png", dpi=200)
    if show:
        plt.show()


def plot_test_prediction(
    test, prediction, title="", fname="tmp.png", ticker="", show=True
):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-plot-test-prediction{ticker}"

    plt.plot(test, lw=0.6, label="real price")
    plt.plot(prediction, lw=0.6, label="predicted price", color="green")

    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.legend()

    plt.savefig(fname, dpi=200)
    if show:
        plt.show()


def plot_metrics(rmse, mse, mae, fname="tmp.png", ticker="", title="", show=True):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-plot-metrics{ticker}"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(rmse, lw=0.6)
    ax1.set(xlabel="#epochs", ylabel="RMSE")

    ax2.plot(mse, lw=0.6)
    ax2.set(xlabel="#epochs", ylabel="MSE")

    ax3.plot(mae, lw=0.6)
    ax3.set(xlabel="#epochs", ylabel="MAE")

    plt.title(title)
    # plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200)
    if show:
        plt.show()


def plot_metrics_val(
    history, rmse, mse, mae, fname="tmp.png", ticker="", title="", show=True
):
    folder = "report/"
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    ticker = f"-{ticker}" if ticker != "" else ticker
    fname = f"{folder}{fname}-plot-metrics{ticker}"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(history.history[rmse], lw=0.6)
    ax1.plot(history.history["val_" + rmse], lw=0.6)
    ax1.legend(["rmse", "val rmse"])
    ax1.set(xlabel="#epochs", ylabel="RMSE")

    ax2.plot(history.history[mse], lw=0.6)
    ax2.plot(history.history["val_" + rmse], lw=0.6)
    ax2.legend([mse, "val " + mse])
    ax2.set(xlabel="#epochs", ylabel="MSE")

    ax3.plot(history.history[mae], lw=0.6)
    ax3.plot(history.history["val_" + mae], lw=0.6)
    ax3.legend([mae, "val " + mae])
    ax3.set(xlabel="#epochs", ylabel="MAE")

    plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=200)
    if show:
        plt.show()


# def plot_metrics_val(history, metric, show=True):
#     plt.plot(history.history[metric])
#     plt.plot(history.history["val_" + metric], "")
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend([metric, "val_" + metric])
