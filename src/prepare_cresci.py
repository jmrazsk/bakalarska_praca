
from pathlib import Path
import pandas as pd

# Cesty k datasetu
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / "cresci-2017"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 0 = genuine account, 1 = bot
CATEGORIES = {
    "genuine_accounts": 0,
    "social_spambots_1": 1,
}

MAX_TWEETS_PER_ACCOUNT = 20


def read_csv_file(path):
    """Načíta CSV aj v prípade, že súbor používa iné kódovanie."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Nepodarilo sa načítať súbor: {path}")


def find_column(df, possible_names):
    """Nájde stĺpec bez ohľadu na veľkosť písmen."""
    columns = {str(c).strip().lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


def prepare_category(category, label):
    folder = RAW_DIR / category
    users_path = folder / "users.csv"
    tweets_path = folder / "tweets.csv"

    if not users_path.exists() or not tweets_path.exists():
        print(f"Preskakujem {category}: chýba users.csv alebo tweets.csv")
        return None

    users = read_csv_file(users_path)
    tweets = read_csv_file(tweets_path)

    print(f"\nSpracovávam: {category}")
    print(f"Počet účtov: {len(users)}")
    print(f"Počet tweetov: {len(tweets)}")

    # Zisti identifikátory účtov v oboch súboroch
    user_id_col = find_column(users, ["id", "user_id", "userid"])
    tweet_user_id_col = find_column(
        tweets, ["user_id", "userid", "user.id", "id"]
    )

    if user_id_col is None or tweet_user_id_col is None:
        print("Nepodarilo sa automaticky nájsť ID stĺpce.")
        print("Stĺpce users.csv:", users.columns.tolist())
        print("Stĺpce tweets.csv:", tweets.columns.tolist())
        raise ValueError(
            "Skontroluj názvy ID stĺpcov a uprav ich v skripte."
        )

    # Zjednoť formát ID, aby sa dali spojiť
    users[user_id_col] = users[user_id_col].astype("string")
    tweets[tweet_user_id_col] = tweets[tweet_user_id_col].astype("string")

    users = users.drop_duplicates(subset=[user_id_col]).copy()
    users = users.rename(columns={user_id_col: "user_id"})

    # Nájdeme text tweetu
    text_col = find_column(tweets, ["text", "full_text", "tweet"])
    if text_col is None:
        print("Stĺpce tweets.csv:", tweets.columns.tolist())
        raise ValueError("V tweets.csv sa nenašiel stĺpec s textom.")

    # Najviac 20 tweetov na účet, aby bol dataset rozumne veľký
    tweets = tweets[
        ["user_id" if "user_id" in tweets.columns else tweet_user_id_col,
         text_col]
    ].copy()

    tweets = tweets.rename(columns={
        tweet_user_id_col: "user_id",
        text_col: "tweet_text"
    })

    tweets["tweet_text"] = tweets["tweet_text"].fillna("").astype(str)
    tweets = tweets[tweets["tweet_text"].str.strip() != ""]

    # Obmedzenie počtu tweetov na účet
    tweets = tweets.groupby("user_id", sort=False).head(
        MAX_TWEETS_PER_ACCOUNT
    )

    # Spojenie textov tweetov do jedného textu na účet
    tweet_texts = (
        tweets.groupby("user_id", sort=False)["tweet_text"]
        .apply(lambda texts: " ".join(texts))
        .rename("tweets_text")
        .reset_index()
    )

    data = users.merge(tweet_texts, on="user_id", how="left")
    data["tweets_text"] = data["tweets_text"].fillna("")

    # Pridanie označenia triedy
    data["label"] = label
    data["label_name"] = "bot" if label == 1 else "human"
    data["source"] = category

    # Vytvorenie textového vstupu z profilu a tweetov
    name_col = find_column(data, ["name"])
    description_col = find_column(data, ["description", "bio"])
    screen_name_col = find_column(data, ["screen_name", "username"])

    profile_parts = []

    for col in [name_col, screen_name_col, description_col]:
        if col is not None:
            profile_parts.append(data[col].fillna("").astype(str))

    if profile_parts:
        data["profile_text"] = profile_parts[0]
        for part in profile_parts[1:]:
            data["profile_text"] += " " + part
    else:
        data["profile_text"] = ""

    data["text"] = (
        data["profile_text"].str.strip()
        + " "
        + data["tweets_text"].str.strip()
    ).str.strip()

    return data


def main():
    all_data = []

    for category, label in CATEGORIES.items():
        result = prepare_category(category, label)
        if result is not None:
            all_data.append(result)

    if not all_data:
        raise RuntimeError(
            f"Nenašli sa žiadne dáta v priečinku {RAW_DIR}"
        )

    combined = pd.concat(all_data, ignore_index=True)

    # Odstránenie záznamov bez použiteľného textu
    combined = combined[combined["text"].str.strip() != ""].copy()

    output_file = OUTPUT_DIR / "cresci_prepared.csv"
    combined.to_csv(output_file, index=False, encoding="utf-8")

    print("\n--- Hotovo ---")
    print(f"Výsledný súbor: {output_file}")
    print(f"Počet účtov: {len(combined)}")
    print("\nPočty podľa tried:")
    print(combined["label_name"].value_counts())
    print("\nVýsledné stĺpce:")
    print(combined.columns.tolist())


if __name__ == "__main__":
    main()
