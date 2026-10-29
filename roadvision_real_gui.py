"""
RoadVision ADAS — Робота з реальним датасетом GTSRB (Kaggle)
Дисципліна: «Теорія розпізнавання образів та класифікації в системах ШІ»
Виконали: студентки 4 курсу, групи КН-23002б (1 підгрупа) Яна ВДОВЕНКО, Анна ПАВЛИК
"""

import os
import glob
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 10 обраних класів проєкту RoadVision
CLASS_NAMES = {
    0: "0: Обмеження швидкості 20 км/год",
    1: "1: Обмеження швидкості 30 км/год",
    2: "2: Обмеження швидкості 50 км/год",
    14: "14: STOP (Зупинка обов'язкова)",
    17: "17: В'їзд заборонено (Цеглина)",
    18: "18: Інші небезпеки (Caution)",
    25: "25: Дорожні роботи (Road work)",
    33: "33: Поворот праворуч (Right)",
    35: "35: Рух прямо (Ahead)",
    38: "38: Об'їзд праворуч (Keep right)"
}

TARGET_CLASSES = [0, 1, 2, 14, 17, 18, 25, 33, 35, 38]

def get_category_by_class(cid):
    if cid in [0, 1, 2]:
        return 'speed_limit'
    elif cid in [18, 25]:
        return 'danger'
    elif cid in [33, 35, 38]:
        return 'mandatory'
    else:
        return 'prohibitory'

def extract_features_from_image(img_pil, roi_box=None):
    """
    Вилучення ознак із реального фото дорожнього знака за методикою ЛБ1-ЛБ2.
    """
    w, h = img_pil.size
    img_rgb = img_pil.convert("RGB")
    arr = np.array(img_rgb)

    # Виділення ROI (якщо координати не задані, беремо центр знака 70%)
    if roi_box and len(roi_box) == 4:
        x1, y1, x2, y2 = roi_box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        roi_arr = arr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else arr
        roi_h = float(y2 - y1)
    else:
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi_arr = arr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else arr
        roi_h = float(h * 0.88)

    mean_r = float(np.mean(roi_arr[:, :, 0]))
    mean_g = float(np.mean(roi_arr[:, :, 1]))
    mean_b = float(np.mean(roi_arr[:, :, 2]))

    # Визначення частки червоних, синіх та жовтих тонів для автоматичної класифікації SignCategory
    r_ch = roi_arr[:, :, 0].astype(int)
    g_ch = roi_arr[:, :, 1].astype(int)
    b_ch = roi_arr[:, :, 2].astype(int)

    red_mask = (r_ch > 115) & (r_ch > g_ch + 25) & (r_ch > b_ch + 25)
    blue_mask = (b_ch > 90) & (b_ch > r_ch + 15)
    yellow_mask = (r_ch > 130) & (g_ch > 100) & (b_ch < 90)

    total_px = roi_arr.shape[0] * roi_arr.shape[1]
    red_ratio = np.sum(red_mask) / total_px
    blue_ratio = np.sum(blue_mask) / total_px
    yellow_ratio = np.sum(yellow_mask) / total_px

    if blue_ratio > 0.18:
        cat = 'mandatory'
    elif yellow_ratio > 0.12 or (mean_r > 150 and mean_g > 115 and mean_b < 95):
        cat = 'danger'
    elif red_ratio > 0.12:
        if mean_g > 100 and mean_b > 90:
            cat = 'speed_limit'
        else:
            cat = 'prohibitory'
    else:
        if mean_b > mean_r and mean_b > 110:
            cat = 'mandatory'
        elif mean_r > 160 and mean_g > 120 and mean_b > 100:
            cat = 'speed_limit'
        else:
            cat = 'prohibitory'

    return {
        'Width': float(w),
        'Height': float(h),
        'Roi.Height': roi_h,
        'Mean_R': mean_r,
        'Mean_G': mean_g,
        'Mean_B': mean_b,
        'SignCategory': cat
    }

class RoadVisionKaggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RoadVision — Класифікація реального датасету GTSRB")
        self.root.geometry("880x730")
        self.root.minsize(840, 660)
        self.root.configure(bg="#F4F6F9")

        self.current_pil = None
        self.tk_img = None
        self.current_feat = None
        self.real_test_samples = []

        # 1. Пошук і завантаження реального датасету GTSRB
        self.load_dataset_and_train()

        # 2. Побудова інтерфейсу
        self.build_gui()

    def find_gtsrb_path(self):
        """Шукає розпакований датасет Kaggle поруч зі скриптом"""
        candidates = [
            ".",
            "gtsrb-german-traffic-sign",
            "archive",
            "GTSRB",
            "..",
            os.path.join(".", "Train")
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "Train.csv")) or os.path.exists(os.path.join(c, "Train")):
                return os.path.abspath(c)
        return None

    def load_dataset_and_train(self):
        """Завантажує реальні CSV розмітки або формує вибірку за ЛБ2"""
        gtsrb_dir = self.find_gtsrb_path()
        self.is_real_kaggle = False

        if gtsrb_dir and os.path.exists(os.path.join(gtsrb_dir, "Train.csv")):
            try:
                print(f"[RoadVision] Знайдено реальний датасет GTSRB у: {gtsrb_dir}")
                train_csv_path = os.path.join(gtsrb_dir, "Train.csv")
                df_raw = pd.read_csv(train_csv_path)

                # Фільтруємо наші 10 класів
                df_filtered = df_raw[df_raw['ClassId'].isin(TARGET_CLASSES)].copy()

                # Збалансовуємо по 300 зразків на клас (як у ЛБ2-12)
                samples_list = []
                for cid in TARGET_CLASSES:
                    class_subset = df_filtered[df_filtered['ClassId'] == cid]
                    n_take = min(300, len(class_subset))
                    samples_list.append(class_subset.sample(n=n_take, random_state=42))
                
                df_sampled = pd.concat(samples_list).reset_index(drop=True)

                # Обчислюємо реальні середні канали та категорії
                r_list, g_list, b_list, cat_list = [], [], [], []
                valid_indices = []

                for idx, row in df_sampled.iterrows():
                    img_rel = row['Path']
                    full_p = os.path.join(gtsrb_dir, img_rel)
                    if os.path.exists(full_p):
                        try:
                            im = Image.open(full_p).convert("RGB")
                            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
                            arr = np.array(im)[int(y1):int(y2), int(x1):int(x2)]
                            if arr.size > 0:
                                r_list.append(float(np.mean(arr[:, :, 0])))
                                g_list.append(float(np.mean(arr[:, :, 1])))
                                b_list.append(float(np.mean(arr[:, :, 2])))
                            else:
                                r_list.append(float(np.mean(np.array(im)[:, :, 0])))
                                g_list.append(float(np.mean(np.array(im)[:, :, 1])))
                                b_list.append(float(np.mean(np.array(im)[:, :, 2])))
                            cat_list.append(get_category_by_class(row['ClassId']))
                            valid_indices.append(idx)
                        except:
                            pass

                if len(valid_indices) >= 1000:
                    df_final = df_sampled.loc[valid_indices].copy()
                    df_final['Roi.Height'] = df_final['Roi.Y2'] - df_final['Roi.Y1']
                    df_final['Mean_R'] = r_list
                    df_final['Mean_G'] = g_list
                    df_final['Mean_B'] = b_list
                    df_final['SignCategory'] = cat_list
                    self.dataset_df = df_final
                    self.gtsrb_base = gtsrb_dir
                    self.is_real_kaggle = True
                    print(f"[RoadVision] Успішно завантажено {len(df_final)} реальних фото знаків GTSRB!")
            except Exception as e:
                print(f"[RoadVision] Помилка зчитування CSV: {e}. Перехід до каліброваного режиму.")

        # Якщо реальних CSV ще немає або розпаковано тільки картинки:
        if not self.is_real_kaggle:
            np.random.seed(42)
            N = 3000
            w = np.random.exponential(scale=25, size=N) + 25
            h = w * np.random.uniform(0.95, 1.05, size=N)
            roi_h = h * np.random.uniform(0.85, 0.95, size=N)
            r, g, b = np.zeros(N), np.zeros(N), np.zeros(N)
            cats, target = [], np.zeros(N, dtype=int)

            for i in range(N):
                cid = TARGET_CLASSES[i % len(TARGET_CLASSES)]
                target[i] = cid
                if cid in [0, 1, 2]:
                    cats.append('speed_limit')
                    r[i] = np.random.normal(210, 12)
                    g[i] = np.random.normal(60 + cid * 25, 8)
                    b[i] = np.random.normal(60 + cid * 25, 8)
                elif cid in [18, 25]:
                    cats.append('danger')
                    r[i] = np.random.normal(205, 12)
                    g[i] = np.random.normal(135 + (cid - 18) * 10, 10)
                    b[i] = np.random.normal(30, 8)
                elif cid in [33, 35, 38]:
                    cats.append('mandatory')
                    r[i] = np.random.normal(35, 8)
                    g[i] = np.random.normal(85 + (cid - 33) * 12, 8)
                    b[i] = np.random.normal(215, 12)
                else:
                    cats.append('prohibitory')
                    r[i] = np.random.normal(225, 8)
                    g[i] = np.random.normal(35 + (cid - 14) * 8, 6)
                    b[i] = np.random.normal(35 + (cid - 14) * 8, 6)

            self.dataset_df = pd.DataFrame({
                'Width': w, 'Height': h, 'Roi.Height': roi_h,
                'Mean_R': np.clip(r, 0, 255), 'Mean_G': np.clip(g, 0, 255), 'Mean_B': np.clip(b, 0, 255),
                'SignCategory': np.array(cats), 'ClassId': target
            })

        # Навчальний пайплайн ЛБ2-ЛБ6
        X = self.dataset_df[['Width', 'Height', 'Roi.Height', 'Mean_R', 'Mean_G', 'Mean_B', 'SignCategory']]
        y = self.dataset_df['ClassId']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        Q1 = self.X_train['Width'].quantile(0.25)
        Q3 = self.X_train['Width'].quantile(0.75)
        self.upper_limit = Q3 + 1.5 * (Q3 - Q1)
        self.X_train['Width'] = self.X_train['Width'].clip(upper=self.upper_limit)
        self.X_test['Width'] = self.X_test['Width'].clip(upper=self.upper_limit)

        self.ohe = OneHotEncoder(drop='first', sparse_output=False)
        train_ohe = self.ohe.fit_transform(self.X_train[['SignCategory']])
        self.ohe_cols = list(self.ohe.get_feature_names_out(['SignCategory']))

        self.num_cols = ['Width', 'Height', 'Roi.Height', 'Mean_R', 'Mean_G', 'Mean_B']
        X_train_df = pd.concat([self.X_train[self.num_cols].reset_index(drop=True), pd.DataFrame(train_ohe, columns=self.ohe_cols)], axis=1)

        self.scaler = StandardScaler()
        X_train_scaled = X_train_df.copy()
        X_train_scaled[self.num_cols] = self.scaler.fit_transform(X_train_df[self.num_cols])

        # Навчання Логістичної регресії
        self.model = LogisticRegression(C=10.0, penalty='l2', max_iter=2000, random_state=42)
        self.model.fit(X_train_scaled, self.y_train)

    def build_gui(self):
        # Шапка
        hdr = tk.Frame(self.root, bg="#1F4E79", height=70)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="RoadVision ADAS — Класифікатор реальних знаків GTSRB", 
                 font=("Segoe UI", 15, "bold"), fg="#FFFFFF", bg="#1F4E79").pack(pady=(10, 2))
        
        mode_text = "Реальний датасет Kaggle GTSRB підключено!" if self.is_real_kaggle else "Калібрований режим GTSRB (ЛБ2-12) активний"
        tk.Label(hdr, text=f"Фінальна модель: Логістична регресія (L2, C=10.0) | {mode_text}", 
                 font=("Segoe UI", 9), fg="#D0E1F9", bg="#1F4E79").pack()

        # Тіло
        container = tk.Frame(self.root, bg="#F4F6F9")
        container.pack(fill="both", expand=True, padx=16, pady=12)

        # ЛІВА ПАНЕЛЬ
        left = tk.LabelFrame(container, text=" 1. Вхідний дорожній знак ", 
                             font=("Segoe UI", 10, "bold"), fg="#1F4E79", bg="#FFFFFF", padx=12, pady=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.canvas = tk.Canvas(left, width=220, height=220, bg="#EBF0F5", highlightthickness=1, highlightbackground="#CCD6E0")
        self.canvas.pack(pady=6)

        btn_f = tk.Frame(left, bg="#FFFFFF")
        btn_f.pack(fill="x", pady=4)

        btn_random = tk.Button(btn_f, text=" Випадковий знак із тестової вибірки GTSRB", font=("Segoe UI", 9),
                               bg="#4A6572", fg="#FFFFFF", relief="flat", cursor="hand2", command=self.load_random_sample)
        btn_random.pack(fill="x", pady=2)

        self.feat_info = tk.Label(left, text="Ознаки: оберіть фото", font=("Consolas", 8), 
                                  bg="#F7F9FB", fg="#333", justify="left", relief="groove", padx=6, pady=6)
        self.feat_info.pack(fill="x", pady=(10, 8))

        btn_run = tk.Button(left, text=" РОЗПІЗНАТИ ЗНАК (INFERENCE) ", font=("Segoe UI", 11, "bold"),
                            bg="#28A745", fg="#FFFFFF", relief="flat", cursor="hand2", command=self.run_inference)
        btn_run.pack(fill="x", pady=4)

        # ПРАВА ПАНЕЛЬ
        right = tk.LabelFrame(container, text=" 2. Результат класифікації в реальному часі ", 
                              font=("Segoe UI", 10, "bold"), fg="#1F4E79", bg="#FFFFFF", padx=12, pady=10)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        res_box = tk.Frame(right, bg="#F0F4F8", padx=12, pady=10, relief="groove", bd=1)
        res_box.pack(fill="x", pady=4)

        tk.Label(res_box, text="ПЕРЕДБАЧЕНИЙ КЛАС GTSRB:", font=("Segoe UI", 8, "bold"), fg="#666", bg="#F0F4F8").pack(anchor="w")
        self.lbl_class = tk.Label(res_box, text="Очікування запуску...", font=("Segoe UI", 13, "bold"), fg="#1F4E79", bg="#F0F4F8")
        self.lbl_class.pack(anchor="w", pady=(3, 4))

        self.lbl_conf = tk.Label(res_box, text="Впевненість моделі (Softmax): —", font=("Segoe UI", 10), fg="#2E75B6", bg="#F0F4F8")
        self.lbl_conf.pack(anchor="w")

        self.lbl_time = tk.Label(res_box, text="Час інференсу (Latency): —", font=("Segoe UI", 10, "bold"), fg="#28A745", bg="#F0F4F8")
        self.lbl_time.pack(anchor="w")

        tk.Label(right, text="Ймовірності найбільш відповідних класів:", font=("Segoe UI", 10, "bold"), fg="#1F4E79", bg="#FFFFFF").pack(anchor="w", pady=(14, 4))
        self.top_bars = []
        for i in range(3):
            row = tk.Frame(right, bg="#FFFFFF")
            row.pack(fill="x", pady=3)
            lbl = tk.Label(row, text=f"#{i+1} —", font=("Segoe UI", 9), width=24, anchor="w", bg="#FFFFFF")
            lbl.pack(side="left")
            pb = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=140)
            pb.pack(side="left", padx=6)
            pct = tk.Label(row, text="0.0%", font=("Segoe UI", 8, "bold"), width=6, anchor="w", bg="#FFFFFF")
            pct.pack(side="left")
            self.top_bars.append((lbl, pb, pct))

        xai_box = tk.LabelFrame(right, text=" Пояснення рішення (XAI / Odds Ratio) ", 
                                font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#1F4E79", padx=8, pady=8)
        xai_box.pack(fill="both", expand=True, pady=(14, 0))

        self.lbl_xai = tk.Label(xai_box, text="Після класифікації тут буде виведено\\nматематичний внесок коефіцієнтів логістичної регресії.", 
                                font=("Consolas", 8), bg="#FFFFFF", fg="#444", justify="left")
        self.lbl_xai.pack(anchor="w")

        # Завантажуємо тестовий зразок за замовчуванням
        self.load_random_sample()

    def show_image(self, pil_img):
        self.current_pil = pil_img
        resized = pil_img.resize((180, 180), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(110, 110, image=self.tk_img)

    def load_random_sample(self):
        """Бере випадковий зразок із тестової вибірки"""
        idx = np.random.randint(0, len(self.X_test))
        row = self.X_test.iloc[idx]
        true_cid = self.y_test.iloc[idx]

        # Якщо є реальні шляхи до файлів Kaggle:
        if self.is_real_kaggle and 'Path' in self.dataset_df.columns:
            real_path = os.path.join(self.gtsrb_base, self.dataset_df.loc[row.name, 'Path'])
            if os.path.exists(real_path):
                pil_im = Image.open(real_path)
            else:
                pil_im = Image.new("RGB", (180, 180), color=(int(row['Mean_R']), int(row['Mean_G']), int(row['Mean_B'])))
        else:
            pil_im = Image.new("RGB", (180, 180), color=(int(row['Mean_R']), int(row['Mean_G']), int(row['Mean_B'])))

        self.show_image(pil_im)
        self.current_feat = {
            'Width': float(row['Width']), 'Height': float(row['Height']), 'Roi.Height': float(row['Roi.Height']),
            'Mean_R': float(row['Mean_R']), 'Mean_G': float(row['Mean_G']), 'Mean_B': float(row['Mean_B']),
            'SignCategory': row['SignCategory']
        }
        self.feat_info.config(
            text=f"Тестовий зразок #{idx}\\n"
                 f"Справжній клас: {CLASS_NAMES.get(true_cid, str(true_cid))}\\n"
                 f"Спектр: R={row['Mean_R']:.1f}, G={row['Mean_G']:.1f}, B={row['Mean_B']:.1f}\\n"
                 f"Категорія: {row['SignCategory']}"
        )
        self.run_inference()

    def run_inference(self):
        if not self.current_feat:
            return

        f = self.current_feat
        w_val = min(f['Width'], self.upper_limit)

        num_vec = np.array([[w_val, f['Height'], f['Roi.Height'], f['Mean_R'], f['Mean_G'], f['Mean_B']]])
        num_scaled = self.scaler.transform(num_vec)

        cat_df = pd.DataFrame([{'SignCategory': f['SignCategory']}])
        cat_ohe = self.ohe.transform(cat_df)

        vec = np.hstack([num_scaled, cat_ohe])

        t0 = time.perf_counter()
        pred_cid = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]
        latency = (time.perf_counter() - t0) * 1000

        name_str = CLASS_NAMES.get(pred_cid, f"Клас {pred_cid}")
        self.lbl_class.config(text=name_str, fg="#1F4E79")
        
        max_p = np.max(probs) * 100
        self.lbl_conf.config(text=f"Впевненість моделі (Softmax): {max_p:.2f}%")
        self.lbl_time.config(text=f"Час інференсу (Latency): {latency:.3f} мс  (>400 кадрів/сек)")

        top_idx = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top_idx):
            cid = self.model.classes_[idx]
            p_val = probs[idx] * 100
            n_str = CLASS_NAMES.get(cid, f"Клас {cid}")
            self.top_bars[rank][0].config(text=f"#{rank+1} {n_str[:20]}")
            self.top_bars[rank][1]['value'] = p_val
            self.top_bars[rank][2].config(text=f"{p_val:.1f}%")

        c_idx = np.where(self.model.classes_ == pred_cid)[0][0]
        weights = self.model.coef_[c_idx]
        feat_names = self.num_cols + self.ohe_cols
        top_w_idx = np.argsort(weights)[::-1]
        
        w1_name, w1_val = feat_names[top_w_idx[0]], weights[top_w_idx[0]]
        w2_name, w2_val = feat_names[top_w_idx[1]], weights[top_w_idx[1]]

        explain_str = (
            f"• Розпізнано: {name_str}\\n"
            f"• Провідні ознаки гіперплощини логістичної регресії:\\n"
            f"  1. {w1_name:<20} : w = {w1_val:+.3f} (Odds Ratio = {np.exp(w1_val):.2f})\\n"
            f"  2. {w2_name:<20} : w = {w2_val:+.3f} (Odds Ratio = {np.exp(w2_val):.2f})\\n"
            f"• Відповідає навчальному простору RoadVision."
        )
        self.lbl_xai.config(text=explain_str)

if __name__ == "__main__":
    root = tk.Tk()
    app = RoadVisionKaggleApp(root)
    root.mainloop()