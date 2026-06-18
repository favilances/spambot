import joblib

model = joblib.load("spam_modeli.joblib")
vektorlestirici = joblib.load("vektorlestirici.joblib")

while True:
    mesaj = input("mesaj gir: ")
    if mesaj.lower() == "q":
        break

    vektor = vektorlestirici.transform([mesaj])
    tahmin = model.predict(vektor)[0]
    olasilik = model.predict_proba(vektor)[0]
    olasilik_yuzde = max(olasilik) * 100

    sinif = "SPAM" if tahmin == "spam" else "NORMAL"
    print(f"Sonuç: {sinif} (%{olasilik_yuzde:.2f})")
    print("-" * 20)
