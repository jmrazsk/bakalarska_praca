
from pathlib import Path
import json
import pandas as pd

# Cesty k datasetu
BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "Unmasking-The-Imposters"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "unmasking-the-imposters"
)

# Vytvorenie výstupného priečinka
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]


def load_json_file(file_path):
    """
    Načíta JSON alebo JSONL súbor.
    Podporuje zoznam objektov aj samostatné JSON objekty
    na jednotlivých riadkoch.
    """

    # Najskôr skúsime klasický JSON
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Jeden záznam alebo objekt obsahujúci záznamy
            for key in ["data", "examples", "records"]:
                if isinstance(data.get(key), list):
                    return data[key]

            return [data]

    except json.JSONDecodeError:
        pass

    # Ak to nebol klasický JSON, skúsime JSONL
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(
                    f"Chyba v {file_path}, riadok "
                    f"{line_number}: {error}"
                )

    return records


def prepare_split(file_path, model_name, split_name):
    """
    Pripraví jeden split datasetu.
    """

    records = load_json_file(file_path)

    if not records:
        print(f"Upozornenie: {file_path} je prázdny.")
        return None

    df = pd.DataFrame(records)

    # Kontrola potrebných stĺpcov
    if "text" not in df.columns or "label" not in df.columns:
        print(f"Chýba text alebo label v súbore: {file_path}")
        print("Dostupné stĺpce:", df.columns.tolist())
        return None

    # Ponecháme iba potrebné údaje
    df = df[["text", "label"]].copy()

    # Odstránenie prázdnych textov
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    # Kontrola a konverzia labelov
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Kontrola platných tried
    df = df[df["label"].isin([0, 1])].copy()

    # Pridanie informácií o pôvode dát
    df["model"] = model_name
    df["split"] = split_name

    # Čitateľné označenie tried
    df["label_name"] = df["label"].map({
        0: "human",
        1: "ai"
    })

    # Odstránenie duplicitných záznamov
    df = df.drop_duplicates(
        subset=["text", "label"]
    ).reset_index(drop=True)

    return df


def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset sa nenašiel: {RAW_DIR}"
        )

    all_models = [
        folder for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    ]

    print(f"Nájdených modelov: {len(all_models)}")

    summary = []

    for model_folder in all_models:

        model_name = model_folder.name

        print(f"\nSpracovávam model: {model_name}")

        model_output = OUTPUT_DIR / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:

            input_file = model_folder / f"{split}.json"

            if not input_file.exists():
                print(f"  Chýba súbor: {input_file.name}")
                continue

            df = prepare_split(
                input_file,
                model_name,
                split
            )

            if df is None or df.empty:
                continue

            output_file = model_output / f"{split}.csv"

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8"
            )

            human_count = (df["label"] == 0).sum()
            ai_count = (df["label"] == 1).sum()

            summary.append({
                "model": model_name,
                "split": split,
                "total": len(df),
                "human": human_count,
                "ai": ai_count
            })

            print(f"  {split}: {len(df)} záznamov")
            print(f"    Human: {human_count}")
            print(f"    AI: {ai_count}")

    # Súhrnná tabuľka
    if summary:
        summary_df = pd.DataFrame(summary)

        summary_file = OUTPUT_DIR / "summary.csv"

        summary_df.to_csv(
            summary_file,
            index=False,
            encoding="utf-8"
        )

        print("\n--- SPRACOVANIE DOKONČENÉ ---")
        print(summary_df.to_string(index=False))
        print(f"\nVýsledky nájdeš v: {OUTPUT_DIR}")

    else:
        print("\nNepodarilo sa spracovať žiadne dáta.")


if __name__ == "__main__":
    main()
