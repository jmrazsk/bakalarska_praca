# Záznam z práce na projekte

Pracovný dokument, z ktorého sa neskôr zostaví report. Zapisuje sa, **čo sa urobilo, prečo a na aké problémy sme narazili**. Po každej práci doplniť nový záznam v časti 5 a aktualizovať tabuľky výsledkov v časti 6.

---

## 1. Cieľ projektu

Menší experimentálny projekt (nie kompletná aplikácia ani celá bakalárska práca). Hlavný dôraz je na natrénovaní modelov, ich porovnaní a vyhodnotení.

Dve časti:

1. **Detekcia bot účtov** (dataset Cresci-17): z údajov o účte a z tweetov rozpoznať, či ide o bota alebo človeka.
2. **Detekcia AI-generovaných textov** (dataset Unmasking the Imposters): rozpoznať, či tweet napísal človek alebo AI.

Na záver jednoduchá aplikácia v Streamlite (ak bude čas) a krátka správa.

Sledované metriky: Accuracy, Precision, Recall, F1-score.

## 2. Datasety

### 2.1 Cresci-17 (boty)

- Použité kategórie: `genuine_accounts` (človek, trieda 0) a `social_spambots_1` (bot, trieda 1).
- Príprava: skript `prepare_cresci.py`. Spája `users.csv` a `tweets.csv`, na účet berie najviac 20 tweetov a spája ich do jedného textu (`tweets_text`).
- Výstup: `data/processed/cresci_prepared.csv`, jeden riadok = jeden účet.

### 2.2 Unmasking the Imposters (AI vs. človek)

- 9 AI modelov: gpt4o, Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct, Qwen2-7B-Instruct, Hermes-2-Pro-Llama-3-8B, OpenHermes-2.5-Mistral-7B, dolphin-2.8-mistral-7b-v02, dolphin-2.9-llama3-8b, dolphin-2.9.2-qwen2-7b.
- Pôvodne po jednom priečinku na model, v každom už hotové `train.csv`, `val.csv`, `test.csv`.
- Pre každý model rovnaké počty: train 3 544 + 3 544, val 443 + 443, test 443 + 443 (AI + človek).
- Výstup po spojení a vyčistení: `data/processed/unmasking_clean.csv` (jeden súbor, stĺpce `text`, `label`, `label_name`, `model`, `split`).

## 3. Postup a rozhodnutia

| Rozhodnutie | Dôvod |
| --- | --- |
| Unmasking: používame pôvodné rozdelenie train/val/test | Dáta boli už rozdelené, nerozdeľujeme znova. |
| Cresci: split robíme **po účtoch** | Aby sa tweety jedného účtu nedostali do trainu aj testu (model by si zapamätal autorov). |
| Poradie experimentov pri botoch: metadáta, text, kombinácia, transformer | Postupné porovnanie, čo prináša ktorá informácia. |
| Kombinácia metadát a textu: jeden spoločný model (TF-IDF + zoškálované metadáta v jednej tabuľke) | Najjednoduchšia cesta, dobre porovnateľná s predošlými experimentmi. |
| Transformer (DistilBERT/RoBERTa) ako experiment, nie ako nutnosť | Zistiť, či sa zložitejší model oplatí oproti TF-IDF. Ak je výsledok podobný, aj to je platný záver. |
| Cross-model test pri AI detekcii | Zistiť, či model generalizuje na AI generátor, ktorý nevidel pri tréningu. |

## 4. Ťažkosti a ako sme ich vyriešili

### 4.1 Unmasking: podozrenie na leak v dátach

- **Problém:** Pri explorácii sa ukázalo 35 745 duplicitných textov a v niektorých modeloch rovnaké texty vo viacerých splitoch.
- **Vysvetlenie:** Takmer všetky duplicity sú očakávané. Ľudských textov je 4 430 a tie isté sa opakujú v každom z 9 modelov (4 430 × 8 = 35 440). Skutočných duplicít bolo asi 300. Prekryvy medzi splitmi v rámci modelu boli malé (väčšinou 1 až 2 texty), len pri OpenHermes viac (34 + 6 + 29), pravdepodobne kvôli opakujúcim sa AI výstupom.
- **Riziko pre cross-model experiment:** Ak by boli ľudské texty rozdelené do splitov inak v každom modeli, ľudské texty z testu jedného modelu by mohli byť v trénovacej časti druhého.
- **Kontrola:** Porovnanie ľudských textov každého modelu s modelom gpt4o po splitoch. Výsledok: splity sú vo všetkých modeloch **zarovnané** (ľudské texty z trainu ostávajú v traine, z testu v teste, mimo diagonály 0).
- **Oprava:** Každý text zostal len v jednom splite (priorita test, potom val, potom train) a zmazali sa duplicity v rámci modelu a splitu. Vyradených bolo **564 z 79 740 riadkov** (asi 0,7 %), najviac pri OpenHermes. Výsledok je uložený v `unmasking_clean.csv`.

### 4.2 Cresci-17: text sa líšil typom, nie obsahom

- **Problém:** V exploračnej analýze mal bot v priemere asi 1 820 znakov textu a človek asi 87. Pole `tweets_text` bolo prázdne pre všetkých 3 474 ľudských účtov, takže ich text bol len profil (meno a popis), kým boty mali spojené tweety. Model by sa naučil "dlhý text = bot" a dosiahol by takmer 100 % bez toho, aby sa naučil niečo o botoch.
- **Príčina 1 (formát ID):** V `tweets.csv` boli `user_id` ako desatinné čísla (`678033.0`), v `users.csv` ako celé čísla. Pridaná funkcia `normalize_id`.
- **Príčina 2 (neúplný súbor):** Po oprave ID mali tweety len 1 083 z 3 474 ľudských účtov. Diagnostika ukázala, že všetkých 1 083 ID z `tweets.csv` sa v `users.csv` našlo, takže nešlo o chybu spájania. Súbor `genuine_accounts/tweets.csv` bol **odseknutý**: na konci obsahoval chybovú hlášku z exportu v phpMyAdmin (`Maximum execution time of 300 seconds exceeded`). Tweety ostatných účtov sa pri exporte nezapísali.
- **Rozhodnutie:** Vyradiť účty bez tweetov. Výsledný dataset: **1 083 ľudí a 991 botov**, čo je dokonca vyvážené.
- **Obmedzenie, ktoré treba uviesť v práci:** Súbor bol odseknutý podľa poradia, takže ide o podskupinu ľudí, nie náhodný výber. Mediány metadát (ľudia s tweetmi vs. bez nich):

| | statuses_count | followers_count | friends_count |
| --- | --- | --- | --- |
| bez tweetov | 5 705 | 319 | 304 |
| s tweetmi | 8 272 | 407 | 366 |

  Rovnaký rád veľkosti, ale ľudia s dostupnými tweetmi sú mierne aktívnejší.

### 4.3 Cresci-17: stĺpce, ktoré prezrádzajú triedu

- Stĺpce `test_set_1`, `test_set_2`, `source`, `label`, `label_name` sú pomocné údaje datasetu, nie vlastnosti účtu. Napríklad `test_set_2` má 991 prázdnych hodnôt, presne toľko, koľko je botov. Ak by boli medzi príznakmi, model by sa ich naučil ako skratku.
- Časové stĺpce `crawled_at`, `timestamp`, `updated` sú podozrivé, lebo obe triedy boli stiahnuté v iný čas (v notebooku sekcia 5 to overuje).
- **Rozhodnutie:** Do metadátového modelu ide len vopred zvolený zoznam: `statuses_count`, `followers_count`, `friends_count`, `favourites_count`, `listed_count` a neskôr odvodené príznaky (pomer followers/friends, vek účtu od jedného pevného dátumu pre všetky účty). Do textového modelu ide len `tweets_text`, nie `text`, lebo ten obsahuje aj `screen_name` (mená botov bývajú náhodné reťazce).

### 4.4 Cresci-17: len jeden typ botov

Dataset obsahuje len `social_spambots_1` (991 účtov). Model sa teda učí rozpoznať tento typ spamboty, nie boty všeobecne. Obmedzenie uviesť v práci. Možné rozšírenie: pridať ďalšie kategórie (napr. `social_spambots_2`, `traditional_spambots_1`) do `CATEGORIES` v `prepare_cresci.py`, ak sú stiahnuté.

### 4.5 Git neukladá prázdne priečinky

Kostra projektu s prázdnymi priečinkami sa po fetchi na druhom počítači nevytvorila, pretože Git sleduje len súbory. Riešenie: súbor `.gitkeep` v každom prázdnom priečinku. Pozor na `.gitignore`: ak sa ignoruje napr. `data/raw/`, treba výnimku (`data/raw/*` a `!data/raw/.gitkeep`). Surové tweety sú v `.gitignore` kvôli veľkosti (súbor má stovky MB).


### 4.6 Cresci-17: vplyv `favourites_count` a ablačná štúdia

- **Zistenie:** Pri porovnaní metadát botov a ľudských účtov sa ukázal výrazný rozdiel v `favourites_count`, teda v počte označení cudzích príspevkov ako obľúbených. Medián bol pri botoch **0**, pri ľudských účtoch **1 492**.
- **Podiel nulových hodnôt:** `favourites_count = 0` malo **91,5 % botov**, ale iba **0,9 % ľudských účtov**. V použitej kategórii `social_spambots_1` teda väčšina botov takmer vôbec nelajkovala cudzie tweety, zatiaľ čo ľudské účty túto funkciu využívali častejšie.
- **Ablácia:** Aby sme zistili, či tento príznak skutočne pomáha modelu, zopakovali sme tréning Random Forest bez `favourites_count`, pri zachovaní rovnakého rozdelenia dát a ostatných nastavení.
- **Výsledok:** Po odstránení `favourites_count` klesla testovacia Accuracy z **97,44 % na 96,79 %** a F1-score z **97,26 % na 96,60 %**. Recall zostal rovnaký (**95,30 %**), zatiaľ čo Precision klesla z **99,30 % na 97,93 %**.
- **Ďalšia ablácia:** Pri odstránení troch príznakov (`favourites_count`, `statuses_count` a `statuses_per_day`) dosiahol Random Forest na teste Accuracy **96,15 %** a F1-score **95,92 %**.
- **Interpretácia:** `favourites_count` modelu pomáha, ale nie je jediným príznakom, na ktorom stojí jeho výkon. Po jeho odstránení zostali výsledky stále vysoké. Rozdiel zároveň naznačuje, že správanie účtov v tomto datasete je užitočným signálom. Treba však uviesť, že ide o konkrétnu skupinu `social_spambots_1`; nedá sa z toho vyvodiť, že všetky bot účty na sociálnych sieťach nelajkujú príspevky.

## 5. Chronologický denník

### 10. 11. 2026


- Vytvorený a spustený notebook `01_exploracia_dat.ipynb`.
- Odhalený a opravený leak v Unmasking (časť 4.1).
- Odhalený problém s textami a chýbajúcimi tweetmi v Cresci-17, upravený `prepare_cresci.py` (časti 4.2 až 4.3).
- Notebook s exploráciou prepísaný na nové súbory `cresci_prepared.csv` a `unmasking_clean.csv`.
- Opravená kostra projektu v Gite (časť 4.5).
- **Ďalší krok:** spustiť novú verziu notebooku, skontrolovať dĺžky `tweets_text` oboch tried, potom `02_baseline_bot_metadata.ipynb`.

### 11. 11. 2026 – prvé experimenty detekcie botov

- Vytvorený a spustený notebook `02_bot_detection.ipynb`.
- Pripravené rozdelenie Cresci-17 na train/validation/test v pomere 70/15/15, stratifikované podľa triedy a vykonané na úrovni účtov. Rozdelenie obsahovalo 1 451 účtov v train, 311 vo validation a 312 v teste.
- Natrénované a porovnané modely založené na metadátach, texte tweetov (TF-IDF + Logistic Regression) a kombinácii metadát s textom.
- Najlepší výsledok podľa F1 na validačnej množine dosiahol textový model TF-IDF + Logistic Regression. Na testovacej množine dosiahol F1 **99,66 %**.
- Zistený výrazný rozdiel v `favourites_count`: nulovú hodnotu malo 91,5 % botov a 0,9 % ľudských účtov.
- Vykonaná ablácia Random Forest bez `favourites_count` a ďalšia ablácia bez troch príznakov. Po odstránení `favourites_count` kleslo testovacie F1 z 97,26 % na 96,60 %.
- **Ďalší krok:** preveriť, či takmer dokonalý výsledok textového modelu nesúvisí s uniformitou tweetov v použitej kategórii botov; následne pokračovať transformerom (DistilBERT) a experimentmi na Unmasking the Imposters.

*(Nový záznam pridať sem.)*

## 6. Výsledky (doplniť po experimentoch)

### 6.1 Detekcia botov (Cresci-17)

| Experiment | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Metadáta – Logistic Regression | 97,12 % | 97,95 % | 95,97 % | 96,95 % |
| Metadáta – Random Forest | 97,44 % | 99,30 % | 95,30 % | 97,26 % |
| Text – TF-IDF + Logistic Regression | 99,68 % | 100,00 % | 99,33 % | 99,66 % |
| Metadáta + text – Logistic Regression | 99,04 % | 100,00 % | 97,99 % | 98,98 % |
| Transformer (DistilBERT) | – | – | – | – |

**Poznámka:** Hodnoty v tabuľke sú z testovacej množiny. Výber najlepšieho modelu sa robil podľa F1 na validačnej množine. Textový model dosiahol na validácii F1 = 100 %, preto je vhodné jeho takmer dokonalý výsledok interpretovať opatrne vzhľadom na uniformitu tweetov v použitej kategórii botov.

#### Ablácia metadátového Random Forest

| Variant | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Všetky metadáta | 97,44 % | 99,30 % | 95,30 % | 97,26 % |
| Bez `favourites_count` | 96,79 % | 97,93 % | 95,30 % | 96,60 % |
| Bez `favourites_count`, `statuses_count` a `statuses_per_day` | 96,15 % | 97,24 % | 94,63 % | 95,92 % |

### 6.2 Detekcia AI textov (Unmasking)

| Experiment | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Baseline (TF-IDF + LR) | | | | |
| Transformer (DistilBERT) | | | | |

### 6.3 Cross-model test

| Tréning na | Testovanie na | Accuracy | F1 |
| --- | --- | --- | --- |
| gpt4o | Mistral-7B-Instruct-v0.2 | | |
| Mistral-7B-Instruct-v0.2 | gpt4o | | |

## 7. Poznámky do reportu (obmedzenia)

- Cresci: podskupina ľudí (1 083 z 3 474) kvôli neúplnému súboru s tweetmi, mierne aktívnejšia než zvyšok.
- Cresci: len jeden typ botov.
- Cresci: tweety obmedzené na 20 na účet a spojené do jedného textu.
- Unmasking: ľudské texty sú vo všetkých modeloch rovnaké, líšia sa len AI texty. Dĺžka AI textov sa medzi modelmi líši (napr. gpt4o je v priemere dlhší), čo môže byť čiastočne skratka pre model.