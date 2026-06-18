import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

veri = pd.read_csv("veri_seti.csv")

veri = veri.rename(columns={"Category": "etiket", "Message": "metin"})

metin_egitim, metin_test, etiket_egitim, etiket_test = train_test_split(
    veri["metin"], veri["etiket"], test_size=0.2, random_state=42
)

vektorlestirici = TfidfVectorizer()
x_egitim = vektorlestirici.fit_transform(metin_egitim)
x_test = vektorlestirici.transform(metin_test)

model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(x_egitim, etiket_egitim)

joblib.dump(model, "spam_modeli.joblib")
joblib.dump(vektorlestirici, "vektorlestirici.joblib")

print("eğitim başarıyla tamamlandı")
