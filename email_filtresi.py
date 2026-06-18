import sys
import os
import subprocess
import email
import joblib

model_yolu = "/etc/spambot/spam_modeli.joblib"
vektor_yolu = "/etc/spambot/vektorlestirici.joblib"

if not os.path.exists(model_yolu):
    sys.exit(0)

model = joblib.load(model_yolu)
vektorlestirici = joblib.load(vektor_yolu)

ham_veri = sys.stdin.read()
eposta = email.message_from_string(ham_veri)

govde = ""
if eposta.is_multipart():
    for parca in eposta.walk():
        if parca.get_content_type() == "text/plain":
            govde = parca.get_payload(decode=True).decode(errors="ignore")
            break
else:
    govde = eposta.get_payload(decode=True).decode(errors="ignore")

if govde.strip():
    vektor = vektorlestirici.transform([govde])
    tahmin = model.predict(vektor)[0]
    
    if tahmin == "spam":
        eposta["X-Spam-Flag"] = "YES"
        eposta["X-Spam-Status"] = "Yes"

gonderici = sys.argv[1]
alici = sys.argv[2:]

sendmail = subprocess.Popen(
    ["/usr/sbin/sendmail", "-i", "-f", gonderici] + alici,
    stdin=subprocess.PIPE
)
sendmail.communicate(input=eposta.as_string().encode())
