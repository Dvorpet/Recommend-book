# Book-Crossing dataset – zjištění z analýzy dat (EDA)

Ověřeno na nahraných CSV, 1. 11. 2026. Skripty: `scripts/eda.py`, `scripts/eda2.py`, `scripts/charts.py`.

## Rozměry a technická kvalita
| Soubor | Řádky | Pozn. |
|---|---:|---|
| Books.csv | 271 360 | 8 sloupců, žádné duplicitní ISBN; chybí žánr, popis i cena |
| Users.csv | 278 858 | Age chybí u 40 %, 1 248 nesmyslných věků (<5 / >100); 173 575 uživatelů nemá žádnou interakci |
| Ratings.csv | 1 149 780 | žádné duplicity user × ISBN, žádné chybějící hodnoty |

- **Join Ratings → Books: jen 90 % interakcí** (1,03 M) má metadata; 70 k ratovaných ISBN v Books chybí. Normalizace ISBN skoro nepomůže (+0,04 p. b.).
- Books: 3 řádky s posunutými sloupci, 4 618 knih s rokem 0, 23 knih s rokem v budoucnosti (data jsou z roku 2004), ~4,9 k titulů s HTML entitami (`&amp;`), ~6,5 k titulů a ~2 k autorů s rozbitým kódováním (mojibake), varianty autorů („Stephen King“ / „STEPHEN KING“).
- **Edice:** 40 675 ISBN sdílí titul i autora s jiným ISBN. Harry Potter and the Sorcerer's Stone má 15 ISBN. Bez sloučení na úroveň díla doporučíme stejnou knihu dvakrát.
- Obálky: URL na images.amazon.com z roku 2004 – z prostředí analýzy nedostupné, dostupnost je potřeba ověřit lokálně; nutný fallback.

## Interakce
- **62 % je implicitních (rating 0)**, 38 % explicitních (1–10). Explicitní hodnocení jsou silně pozitivní (průměr 7,6, medián 8) – hodnocení 1–4 tvoří jen 1,7 % explicitních.
- **Extrémní řídkost:** 105 k aktivních uživatelů × 341 k ISBN, hustota 0,003 %.
- **Dlouhý chvost:** 56 % uživatelů a 58 % knih má právě 1 interakci. Medián je 1 na uživatele i na knihu.
- **Power users:** 117 uživatelů s >1 000 interakcemi tvoří 19 % všech dat; top 1 % uživatelů = 48 % interakcí. Několik z nich má ~100 % implicitních záznamů (pravděpodobně hromadné katalogové nahrávky, ne čtení).
- Mezi uživateli s metadaty má ≥5 interakcí jen 20,4 k (22 %); 51 k má právě jednu.
- 5-core (≥5 interakcí na uživatele i knihu): 565 k interakcí, 15,2 k uživatelů, 36 k knih, hustota 0,1 %. 10-core: 6,6 k uživatelů × 13,2 k knih.

## Popularity trap
- Nejčastější kniha je **Wild Animus** (2 502 interakcí, 77 % implicitních, průměr explicitních hodnocení jen 4,4). Šlo o masivně rozdávaný propagační titul. Baseline podle počtu by ji doporučoval všem.
- Prostý průměr hodnocení zase vynáší knihy s několika hodnoceními 10/10.
- Bayesovský průměr (m = 50) dává rozumný žebříček (Harry Potter, Tolkien, To Kill a Mockingbird), ale bez sloučení edic se stejné dílo v top 10 objevuje dvakrát.

## Co z toho plyne pro řešení
1. **Většina zákazníků je cold-start.** Personalizace „pro vás“ dává smysl jen pro menšinu (~20 k uživatelů). Pro ostatní je klíčový item-to-item („podobné knihy“ na detailu knihy) a kvalitní popularita.
2. **Baseline = popularita vážená kvalitou** (Bayesovský průměr), ne prostý počet.
3. **Interakce modelovat jako implicitní signál** (jakákoli interakce = zájem), explicitní hodnocení použít jako váhu. Hodnocení 1–4 jsou vzácná, takže explicitní model hodnocení (RMSE) moc nedává smysl.
4. **Povinné čištění:** filtr na join s Books, sloučení edic na úroveň díla, oprava kódování a HTML entit, omezení vlivu power users (cap / váhování), k-core filtr pro trénink CF.
5. **Content-based je omezený** – máme jen autora, vydavatele, rok a název, žádný žánr ani popis. Pro produkci doporučit obohacení metadat (např. žánry, anotace z katalogu klienta nebo externích zdrojů).
6. **Evaluace bez timestampů:** náhodný per-user holdout (leave-k-out) na 5-core; omezení: nelze simulovat „předpověď budoucnosti“, metriky budou optimistické.
7. **Co tato data neprokážou:** dopad na tržby. Chybí ceny, nákupy, sessions, kliky, čas. Prokázat byznysový přínos lze až na datech klienta a A/B testem.
