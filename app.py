from unittest import result
from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, base64, random, warnings

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ===================== أدوات بيانات =====================

def safe_read_csv(file_storage):
    try_encodings = ["utf-8", "cp1256", "latin-1"]
    for enc in try_encodings:
        try:
            file_storage.stream.seek(0)
            df = pd.read_csv(file_storage, encoding=enc)
            return df
        except Exception:
            continue
    file_storage.stream.seek(0)
    df = pd.read_csv(file_storage, sep=None, engine="python")
    return df

def impute_missing(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())
        else:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                df[c] = df[c].fillna(fill_val)
    return df

def detect_target_column(df):
    common_targets = ["target", "label", "class", "y", "outcome", "diagnosis"]
    for name in df.columns:
        if name.strip().lower() in common_targets:
            return name
    candidate = None
    best_card = 1e9
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= 10:
            if n_unique < best_card:
                best_card = n_unique
                candidate = col
    return candidate if candidate is not None else df.columns[-1]

def encode_features(X):
    # One-Hot للفئات + إبقاء الأرقام
    return pd.get_dummies(X, drop_first=True)

def split_scale(X_enc, y_enc, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=random_state,
        stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test

def summarize_dataframe(df):
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_perc": (df.isna().mean() * 100).round(2).to_dict(),
    }
    desc_num = df.select_dtypes(include=[np.number]).describe().T
    desc_num_html = desc_num.to_html(classes="data-table", border=0)
    return info, desc_num_html

# ===================== الرسومات =====================

def plot_heatmap(df):
    """خريطة الارتباطات بين الميزات الرقمية"""
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(7,5))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _fig_to_b64(dpi=120, bbox=True):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if bbox else None)
    buf.seek(0)
    img64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img64

def plot_top_numeric_distributions(df, max_cols=4):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None
    top = num_cols[:max_cols]
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(top, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[c].dropna(), kde=True, bins=30, color="#0B3954")
        plt.title(c)
    plt.tight_layout()
    return _fig_to_b64()

def plot_target_distribution(y, target_name):
    plt.figure(figsize=(5, 4))
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        sns.histplot(y.dropna(), kde=True, bins=30, color="#8E6C88")
    else:
        sns.countplot(x=y.astype(str), color="#8E6C88")
        plt.xticks(rotation=20, ha="right")
    plt.title(f"Distribution of {target_name}")
    plt.tight_layout()
    return _fig_to_b64()

def make_bar_plot(performance_dict, title):
    plt.figure(figsize=(6, 4))
    names = list(performance_dict.keys())
    vals = list(performance_dict.values())
    sns.barplot(x=names, y=vals, palette="coolwarm")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    return _fig_to_b64()

def make_line_plot(xs, ys, title, xlabel, ylabel):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=xs, y=ys, marker="o", color="seagreen")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    return _fig_to_b64()

# ===================== الخوارزمية الجينية لانتقاء الميزات =====================

# Value Encoding (بتات 0/1)
def init_population(n_individuals, n_features, rng=None):
    rng = rng or random
    pop = []
    for _ in range(n_individuals):
        chrom = [rng.choice([0, 1]) for _ in range(n_features)]
        if not any(chrom):
            chrom[rng.randrange(n_features)] = 1
        pop.append(chrom)
    return pop

def build_model(name, random_state=42):
    name = (name or "logistic").lower()
    if name in ["logistic", "lr", "logreg"]:
        return LogisticRegression(max_iter=2000)
    if name in ["lda", "linear discriminant analysis"]:
        return LinearDiscriminantAnalysis()
    if name in ["tree", "decisiontree", "decision tree"]:
        return DecisionTreeClassifier(random_state=random_state)
    if name in ["rf", "randomforest", "random forest"]:
        return RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    if name in ["svm", "svm_linear", "svm-linear", "linear_svm"]:
        return SVC(kernel="linear", C=1.0, probability=False)
    # افتراضي
    return LogisticRegression(max_iter=2000)

def fitness_of(chrom, df, feature_cols, y, model_name="logistic", random_state=42):
    selected = [f for f, b in zip(feature_cols, chrom) if b == 1]
    if not selected:
        return 0.0

    X = df[selected].copy()
    X = impute_missing(X)

    # ترميز الميزات
    X_enc = encode_features(X)
    # احتفاظ بالأعمدة العددية فقط بعد الترميز
    X_enc = X_enc.select_dtypes(include=[np.number])
    if X_enc.shape[1] == 0:
        return 0.0
    if not np.isfinite(X_enc.values).all():
        return 0.0

    # ترميز الهدف
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    if y_series.dtype == "object" or not pd.api.types.is_numeric_dtype(y_series):
        y_enc, _ = pd.factorize(y_series)
    else:
        y_enc = y_series.values

    # يجب توافر صنفين على الأقل
    if len(np.unique(y_enc)) < 2:
        return 0.0

    # تقسيم وتقييس
    try:
        X_train_sc, X_test_sc, y_train, y_test = split_scale(X_enc, y_enc, random_state=random_state)
    except ValueError:
        return 0.0

    model = build_model(model_name, random_state=random_state)
    try:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        return float(accuracy_score(y_test, y_pred))
    except Exception:
        # fallback دون تقييس إذا لزم
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y_enc, test_size=0.3, random_state=random_state,
                stratify=y_enc if len(np.unique(y_enc)) > 1 else None
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return float(accuracy_score(y_test, y_pred))
        except Exception:
            return 0.0

# Roulette Wheel Selection
def roulette_wheel_selection(pop, fitnesses, rng=None):
    rng = rng or random
    fits = np.array(fitnesses, dtype=float)
    total = fits.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(pop)) / len(pop)
    else:
        probs = fits / total
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return pop[i][:]
    return pop[-1][:]

def single_point_crossover(p1, p2, cx_rate=0.9, rng=None):
    rng = rng or random
    if rng.random() > cx_rate or len(p1) < 2:
        return p1[:], p2[:]
    point = rng.randrange(1, len(p1))
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    if not any(c1):
        c1[rng.randrange(len(c1))] = 1
    if not any(c2):
        c2[rng.randrange(len(c2))] = 1
    return c1, c2

def mutate(chrom, mut_rate=0.02, rng=None):
    rng = rng or random
    for i in range(len(chrom)):
        if rng.random() < mut_rate:
            chrom[i] = 1 - chrom[i]
    if not any(chrom):
        chrom[rng.randrange(len(chrom))] = 1
    return chrom

def genetic_feature_selection(
    df, target_col, pop_size=20, generations=25,
    cx_rate=0.9, mut_rate=0.02, elitism=2, random_state=42,
    selection_method="roulette",  # "roulette" أو "tournament"
    model_name="logistic"         # نموذج الملاءمة داخل GA
):
    rng = random.Random(random_state)
    y = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    if n_features == 0:
        return [], float("nan"), []

    pop = init_population(pop_size, n_features, rng=rng)

    history = []
    best_chrom = None
    best_fit = -1.0

    for gen in range(1, generations + 1):
        fitnesses = [
            fitness_of(chrom, df, feature_cols, y, model_name=model_name, random_state=random_state)
            for chrom in pop
        ]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        gen_best_features = [f for f, b in zip(feature_cols, pop[gen_best_idx]) if b == 1]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_chrom = pop[gen_best_idx][:]

        stats = {
            "AVG": float(np.mean(fitnesses)),
            "STD": float(np.std(fitnesses)),
            "MIN": float(np.min(fitnesses)),
            "MAX": float(np.max(fitnesses))
        }
        history.append({
            "generation": gen,
            "score": round(gen_best_fit, 4),
            "AVG": round(stats["AVG"], 4),
            "STD": round(stats["STD"], 4),
            "MIN": round(stats["MIN"], 4),
            "MAX": round(stats["MAX"], 4),
            "features": gen_best_features
        })

        elite_indices = list(np.argsort(fitnesses))[::-1][:elitism]
        new_pop = [pop[i][:] for i in elite_indices]

        def select_one():
            if selection_method == "roulette":
                return roulette_wheel_selection(pop, fitnesses, rng=rng)
            # tournament fallback
            k = min(3, len(pop))
            idxs = [rng.randrange(len(pop)) for _ in range(k)]
            best_idx = max(idxs, key=lambda i: fitnesses[i])
            return pop[best_idx][:]

        while len(new_pop) < pop_size:
            p1 = select_one()
            p2 = select_one()
            c1, c2 = single_point_crossover(p1, p2, cx_rate=cx_rate, rng=rng)
            c1 = mutate(c1, mut_rate=mut_rate, rng=rng)
            if len(new_pop) < pop_size:
                new_pop.append(c1)
            c2 = mutate(c2, mut_rate=mut_rate, rng=rng)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

    best_features = [f for f, b in zip(feature_cols, best_chrom) if b == 1] if best_chrom else []
    return history, float(best_fit), best_features

# ===================== Chi2 Feature Selection =====================
def chi2_feature_selection(df, target_col, k=10):
    """اختيار أفضل k ميزات باستخدام اختبار Chi-square"""
    # فصل الهدف عن الميزات
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # تعويض القيم المفقودة
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)

    # ترميز المتغيرات الفئوية
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # فقط الميزات الرقمية
    X = X.select_dtypes(include='number')

    # التأكد من عدم وجود NaN بعد الترميز
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # مقياس القيم بين 0 و 1 لأن chi2 يحتاج قيم موجبة
    X_scaled = MinMaxScaler().fit_transform(X)

    # تطبيق اختبار Chi2
    chi2_selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
    chi2_selector.fit(X_scaled, y)
    scores = chi2_selector.scores_
    feature_names = X.columns

    # رسم الأهمية
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, scores, color='skyblue')
    plt.title("Chi² Feature Importance")
    plt.xlabel("Score")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    chi_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    top_features = feature_names[np.argsort(scores)[::-1][:k]].tolist()
    return top_features, scores, chi_plot64


# ===================== Lasso / Logistic L1 Feature Selection =====================
def lasso_feature_selection(df, target_col):
    """اختيار الميزات باستخدام Logistic Regression مع عقوبة L1 (Lasso)"""
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # تعويض القيم المفقودة
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)

    # ترميز الفئات
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # إزالة القيم غير المنتهية أو المفقودة
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # توحيد القيم (مهم للـ L1)
    X_scaled = StandardScaler().fit_transform(X)

    # تدريب نموذج Logistic L1
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
    model.fit(X_scaled, y)

    coef = np.abs(model.coef_[0])
    importance = pd.Series(coef, index=X.columns).sort_values(ascending=False)

    # رسم أعلى الميزات تأثيراً
    plt.figure(figsize=(8, 4))
    importance.head(15).plot(kind='barh', color='purple')
    plt.title("Lasso (L1) Feature Coefficients")
    plt.xlabel("|Coefficient|")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    lasso_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    top_features = importance.head(10).index.tolist()
    return top_features, coef, lasso_plot64

# =====================  الطرق التقليدية =====================

def train_baselines(df, target_col, random_state=42):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
        y_enc, _ = pd.factorize(y)
    else:
        y_enc = y.values if isinstance(y, pd.Series) else y

    if len(np.unique(y_enc)) < 2:
        return {
            "Logistic Regression": float("nan"),
            "Random Forest": float("nan"),
            "Gradient Boosting": float("nan"),
        }

    X = impute_missing(X)
    X_enc = encode_features(X)

    stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.3, random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
       "Random Forest": RandomForestClassifier(n_estimators=80, random_state=random_state),
       "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
       "Logistic Regression": LogisticRegression(max_iter=500)
    }
    performance = {}
    for name, model in models.items():
        try:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            performance[name] = float(accuracy_score(y_test, y_pred))
        except Exception:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance[name] = float(accuracy_score(y_test, y_pred))
            except Exception:
                performance[name] = float("nan")

    return performance

# =====================  F1 & Precision  =====================
def evaluate_classification_metrics(df, target_col):

    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]))
    X = X.select_dtypes(include='number')
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    metrics_df = pd.DataFrame(report).T.round(3)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    conf_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return metrics_df.to_html(classes='table table-bordered table-sm text-center', border=0), conf_plot64

# ===================== PCA =====================

def run_pca(df, target_col):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]))

    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)
    
    X = X.select_dtypes(include='number')
    X_scaled = StandardScaler().fit_transform(X)
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)  # حماية إضافية ضد NaN

    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    components = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    # مخطط التباين
    plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(var_ratio)*100, marker='o')
    plt.title("Cumulative Explained Variance (%)")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_var64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2D
    plt.figure(figsize=(6,5))
    plt.scatter(components[:,0], components[:,1], c=pd.factorize(y)[0], cmap='tab10')
    plt.title("PCA 2D Scatter")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_2d64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 3D
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(components[:,0], components[:,1], components[:,2], c=pd.factorize(y)[0], cmap='tab10')
    ax.set_title("PCA 3D Visualization")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_3d64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    info = {
        "explained_variance": np.round(var_ratio * 100, 2).tolist(),
        "total_variance": round(np.sum(var_ratio) * 100, 2)
    }
    return info, pca_var64, pca_2d64, pca_3d64

# ===================== المسارات =====================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("csv_file")
    if not file:
        return "<h2> لم يتم استلام ملف</h2>"

    try:
        df = safe_read_csv(file)
    except Exception as e:
        return f"<h2> خطأ في قراءة الملف: {e}</h2>"

    if df.empty or df.shape[1] < 2:
        return "<h2> الملف فارغ أو لا يحتوي ميزات كافية</h2>"

    df = impute_missing(df)

    info, desc_num_html = summarize_dataframe(df)

    target_hint = request.form.get("target_hint", "").strip()
    target_col = target_hint if target_hint and target_hint in df.columns else detect_target_column(df)

    rows, cols = df.shape
    num_cols_count = df.select_dtypes(include=[np.number]).shape[1]
    cat_cols_count = df.select_dtypes(exclude=[np.number]).shape[1]
    preview_html = df.head(200).to_html(classes="data-table", index=False, border=0)
    hist_num64 = plot_top_numeric_distributions(df)
    target_dist64 = plot_target_distribution(df[target_col], target_col)

    # إعدادات GA من النموذج (يمكنك إضافة حقول في الواجهة للخيارات)
    def get_num(name, default, t=float):
        val = request.form.get(name, str(default))
        try:
            return t(val)
        except Exception:
            return default

    pop_size = int(get_num("pop_size", 20, int))
    generations = int(get_num("generations", 25, int))
    cx_rate = float(get_num("cx_rate", 0.9, float))
    mut_rate = float(get_num("mut_rate", 0.02, float))
    elitism = int(get_num("elitism", 2, int))


    # خيار اختيار النموذج داخل GA
    ga_model_name = (request.form.get("ga_model", "logistic") or "logistic").strip()

    # حراسة القيم
    pop_size = max(4, min(pop_size, 200))
    generations = max(5, min(generations, 200))
    cx_rate = min(max(cx_rate, 0.0), 1.0)
    mut_rate = min(max(mut_rate, 0.0), 1.0)
    elitism = max(0, min(elitism, max(1, pop_size // 2)))

    # تشغيل الخوارزمية الجينية
    ga_history, ga_best, ga_best_features = genetic_feature_selection(
        df, target_col, pop_size=pop_size, generations=generations,
        cx_rate=cx_rate, mut_rate=mut_rate, elitism=elitism,
        random_state=42, selection_method="roulette",
        model_name=ga_model_name
    )

      # خريطة الارتباطات
    heatmap64 = plot_heatmap(df) 

    # نماذج الأساس
    performance = train_baselines(df, target_col)
    perf_for_plot = dict(performance)
    perf_for_plot["Genetic Algorithm (Best)"] = ga_best

     # Chi-square
    chi_features, chi_scores, chi_plot64 = chi2_feature_selection(df, target_col)

    # Lasso / Logistic L1
    lasso_features, lasso_scores, lasso_plot64 = lasso_feature_selection(df, target_col)

    # PCA
    pca_info, pca_var64, pca_2d64, pca_3d64 = run_pca(df, target_col)
     
    # تقارير الأداء
    class_metrics, conf64 = evaluate_classification_metrics(df, target_col)

    # رسم المقارنة
    perf_for_plot = dict(performance)
    perf_for_plot["Genetic Algorithm (Best)"] = ga_best
    graph_models64 = make_bar_plot(perf_for_plot, "Comparison of Models vs Genetic Algorithm")

    # خط تطور دقة GA عبر الأجيال
    ga_scores = [h["score"] for h in ga_history] if ga_history else []
    ga_line64 = make_line_plot(
        xs=list(range(1, len(ga_scores) + 1)),
        ys=ga_scores,
        title="GA Best Score per Generation",
        xlabel="Generation",
        ylabel="Accuracy",
    ) if ga_scores else None

    return render_template(
        "result.html",
        rows=rows, cols=cols, target=target_col,
        columns=info["columns"], dtypes=info["dtypes"], missing_perc=info["missing_perc"],
        desc_num_html=desc_num_html,
        num_cols=num_cols_count, cat_cols=cat_cols_count,
        preview=preview_html,
        hist_num=hist_num64,
        target_dist=target_dist64,
        generations=ga_history,
        best_score=ga_best,
        best_features=ga_best_features,
        performance=performance,
        graph_models=graph_models64,
        ga_line=ga_line64,
        chi_features=chi_features, chi_plot=chi_plot64,
        lasso_features=lasso_features, lasso_plot=lasso_plot64,
        pca_info=pca_info, pca_var=pca_var64, pca_2d=pca_2d64, pca_3d=pca_3d64,
        class_metrics=class_metrics, conf_plot=conf64,
        heatmap=heatmap64,
    )


@app.route("/download_features", methods=["POST"])
def download_features():
    features_str = request.form.get("features", "")
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    features_df = pd.DataFrame(features, columns=["Selected Features"])
    csv = features_df.to_csv(index=False, encoding='utf-8-sig')
    response = make_response(csv)
    response.headers["Content-Disposition"] = "attachment; filename=selected_features.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)