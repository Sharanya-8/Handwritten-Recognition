import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import threading

model = None


MODEL_PATH = "digit_cnn.keras"

def train_model():
    global model
    import os
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import mnist

    if os.path.exists(MODEL_PATH):
        status_var.set("Loading saved model...")
        model = models.load_model(MODEL_PATH)
        status_var.set("Ready — draw a digit")
        predict_btn.config(state="normal")
        return

    status_var.set("Loading MNIST...")
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train[..., np.newaxis]

    status_var.set("Training CNN, please wait...")
    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1)
    m.save(MODEL_PATH)
    model = m
    status_var.set("Ready — draw a digit")
    predict_btn.config(state="normal")

# ── UI constants ──────────────────────────────────────────────
CANVAS_SIZE = 300
BG       = "#1e1e2e"
SURFACE  = "#181825"
ACCENT   = "#cba6f7"
ACCENT2  = "#89b4fa"
MUTED    = "#6c7086"
FG       = "#cdd6f4"
BTN_BG   = "#313244"

root = tk.Tk()
root.title("Digit Recognizer")
root.configure(bg=BG)
root.resizable(False, False)

# ── Fonts ─────────────────────────────────────────────────────
F_TITLE  = tkfont.Font(family="Segoe UI", size=17, weight="bold")
F_LABEL  = tkfont.Font(family="Segoe UI", size=11)
F_BTN    = tkfont.Font(family="Segoe UI", size=10, weight="bold")
F_PRED   = tkfont.Font(family="Segoe UI", size=64, weight="bold")
F_CONF   = tkfont.Font(family="Segoe UI", size=11)

# ── Title ─────────────────────────────────────────────────────
tk.Label(root, text="✏  Digit Recognizer", font=F_TITLE,
         bg=BG, fg=ACCENT).pack(pady=(20, 6))

# ── Canvas card ───────────────────────────────────────────────
card = tk.Frame(root, bg=ACCENT, padx=2, pady=2, relief="flat")
card.pack(padx=28)

inner = tk.Frame(card, bg=SURFACE)
inner.pack()

canvas = tk.Canvas(inner, width=CANVAS_SIZE, height=CANVAS_SIZE,
                   bg=SURFACE, cursor="crosshair", highlightthickness=0)
canvas.pack()

# Draw a faint guide box to encourage centered, proportional drawing
def draw_guide():
    m = 40
    canvas.create_rectangle(m, m, CANVAS_SIZE-m, CANVAS_SIZE-m,
                             outline="#2a2a3e", width=1, dash=(4, 4), tags="guide")
canvas.after(100, draw_guide)

pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw_ctx = ImageDraw.Draw(pil_img)

# ── Prediction area ───────────────────────────────────────────
pred_frame = tk.Frame(root, bg=BG)
pred_frame.pack(pady=(10, 0))

pred_var = tk.StringVar(value="?")
tk.Label(pred_frame, textvariable=pred_var, font=F_PRED,
         bg=BG, fg=ACCENT, width=3).pack()

conf_var = tk.StringVar(value="")
tk.Label(pred_frame, textvariable=conf_var, font=F_CONF,
         bg=BG, fg=ACCENT2).pack()

tk.Label(root, text="predicted digit", font=F_LABEL,
         bg=BG, fg=MUTED).pack(pady=(0, 6))

# ── Status ────────────────────────────────────────────────────
status_var = tk.StringVar(value="Initializing...")
tk.Label(root, textvariable=status_var, font=F_LABEL,
         bg=BG, fg=MUTED).pack(pady=(0, 10))

# ── Buttons ───────────────────────────────────────────────────
btn_row = tk.Frame(root, bg=BG)
btn_row.pack(pady=(0, 22))

def btn(parent, text, cmd, **kw):
    return tk.Button(parent, text=text, command=cmd, font=F_BTN,
                     bg=BTN_BG, fg=FG, activebackground=ACCENT,
                     activeforeground=BG, relief="flat",
                     padx=20, pady=9, cursor="hand2",
                     borderwidth=0, **kw)

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(img):
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=2)))
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    cropped = arr[r0:r1+1, c0:c1+1]
    h, w = cropped.shape
    size = max(h, w) + 28
    sq = np.zeros((size, size), dtype=np.uint8)
    sq[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = cropped
    resized = np.array(Image.fromarray(sq).resize((28, 28), Image.LANCZOS))
    return resized.astype("float32") / 255.0

def predict():
    if model is None:
        return
    arr = preprocess(pil_img)
    if arr is None:
        return
    inp = arr.reshape(1, 28, 28, 1)
    probs = model.predict(inp, verbose=0)[0]
    digit = int(np.argmax(probs))
    conf  = float(probs[digit]) * 100
    pred_var.set(str(digit))
    conf_var.set(f"{conf:.1f}% confidence")

def clear():
    canvas.delete("all")
    draw_ctx.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    pred_var.set("?")
    conf_var.set("")
    draw_guide()

predict_btn = btn(btn_row, "Predict  (P)", predict, state="disabled")
predict_btn.pack(side="left", padx=5)
btn(btn_row, "Clear  (C)", clear).pack(side="left", padx=5)
btn(btn_row, "Quit  (Q)", root.quit).pack(side="left", padx=5)

# ── Drawing ───────────────────────────────────────────────────
last = [None, None]

def on_press(e):
    last[0], last[1] = e.x, e.y
    canvas.delete("guide")

def on_drag(e):
    r = 8
    lx, ly = last[0], last[1]
    if lx is not None:
        steps = max(abs(e.x - lx), abs(e.y - ly)) // 3 + 1
        for i in range(steps + 1):
            t = i / steps
            ix = int(lx + t * (e.x - lx))
            iy = int(ly + t * (e.y - ly))
            canvas.create_oval(ix-r, iy-r, ix+r, iy+r,
                               fill="white", outline="")
            draw_ctx.ellipse([ix-r, iy-r, ix+r, iy+r], fill=255)
    last[0], last[1] = e.x, e.y

def on_release(e):
    last[0] = last[1] = None

canvas.bind("<ButtonPress-1>", on_press)
canvas.bind("<B1-Motion>", on_drag)
canvas.bind("<ButtonRelease-1>", on_release)

root.bind("p", lambda e: predict())
root.bind("c", lambda e: clear())
root.bind("q", lambda e: root.quit())

# ── Train ─────────────────────────────────────────────────────
threading.Thread(target=train_model, daemon=True).start()

root.mainloop()
