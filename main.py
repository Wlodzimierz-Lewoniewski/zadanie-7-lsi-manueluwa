import numpy as np
import re

def tokenizuj(tekst):
    return [slowo.strip() for slowo in re.split(r'[,\.\?!: ]+', tekst.lower()) if slowo]

n = int(input())
dokumenty = [input().strip() for _ in range(n)]
zapytanie = input().strip()
k = int(input())

slowa_zapytanie = tokenizuj(zapytanie)
tokeny = [tokenizuj(dok) for dok in dokumenty]

unikalne_tokeny = []
for x in tokeny:
    for token in x:
        if token not in unikalne_tokeny:
            unikalne_tokeny.append(token)
unikalne_tokeny = set(unikalne_tokeny)


macierz_incydencji = []
for token in unikalne_tokeny:
    wiersz = []
    for x in tokeny:
        if token in x:
            wiersz.append(1)
        else:
            wiersz.append(0)
    macierz_incydencji.append(wiersz)

macierz_incydencji = np.array(macierz_incydencji)


U, sigma, VT = np.linalg.svd(macierz_incydencji.T, full_matrices=False)

U_k = U[:, :k]
Sigma_k = np.diag(sigma[:k])
VT_k = VT[:k, :]

sprawdz_zapytanie = []
for token in unikalne_tokeny:
    if token in slowa_zapytanie:
        sprawdz_zapytanie.append(1)
    else:
        sprawdz_zapytanie.append(0)

sprawdz_zapytanie = np.array(sprawdz_zapytanie)

q_k = sprawdz_zapytanie @ VT_k.T @ np.linalg.inv(Sigma_k)


podobienstwa = []
for wektor_dokumentu in (U_k @ Sigma_k):
    iloczyn = np.dot(q_k, wektor_dokumentu)
    norma_q_k = np.linalg.norm(q_k)
    norma_wektora = np.linalg.norm(wektor_dokumentu)
    podobienstwo = iloczyn / (norma_q_k * norma_wektora)
    podobienstwa.append(round(float(podobienstwo), 2))


print(podobienstwa)
