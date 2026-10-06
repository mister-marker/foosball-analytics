"""
Скрипт-чистильщик меток (Label Sanitizer).

Проблема: разные датасеты с Roboflow используют разные индексы классов.
  - В одном: 0=ball, 1=player
  - В другом: 0=table, 1=player, 2=ball

Решение: читает data.yaml, находит индекс 'ball',
перезаписывает все .txt — оставляет только мяч как class_id=0.

Использование:
    python scripts/clean_labels.py --dataset_dir /path/to/roboflow_dataset
    python scripts/clean_labels.py --dataset_dir /path/to/dataset --ball_class_id 2
"""

import os
import yaml
import argparse
from pathlib import Path


def find_ball_class_id(data_yaml_path: Path) -> int:
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if isinstance(names, dict):
        names_list = [(k, v.lower()) for k, v in names.items()]
    else:
        names_list = [(i, n.lower()) for i, n in enumerate(names)]

    print(f"  Классы в датасете: {[n for _, n in names_list]}")

    ball_keywords = ["ball", "foosball", "soccer_ball", "soccerball"]
    for idx, name in names_list:
        if any(kw in name for kw in ball_keywords):
            print(f"  Найден класс мяча: '{name}' (индекс {idx})")
            return idx

    raise ValueError(
        f"Класс 'ball' не найден! Классы: {[n for _, n in names_list]}\n"
        "Укажи индекс вручную через --ball_class_id"
    )


def clean_label_file(label_path: Path, ball_class_id: int):
    with open(label_path, "r") as f:
        lines = f.readlines()

    kept = []
    removed = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if int(parts[0]) == ball_class_id:
            parts[0] = "0"
            kept.append(" ".join(parts) + "\n")
        else:
            removed += 1

    with open(label_path, "w") as f:
        f.writelines(kept)

    return len(kept), removed


def clean_dataset(dataset_dir: Path, ball_class_id: int = None):
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml не найден в {dataset_dir}")

    if ball_class_id is None:
        ball_class_id = find_ball_class_id(data_yaml)

    label_dirs = list(dataset_dir.rglob("labels"))
    if not label_dirs:
        print("Папки 'labels' не найдены!")
        return

    total_kept, total_removed, files_processed = 0, 0, 0

    for label_dir in label_dirs:
        txt_files = list(label_dir.glob("*.txt"))
        print(f"\nОбрабатываем: {label_dir} ({len(txt_files)} файлов)")
        for txt_file in txt_files:
            kept, removed = clean_label_file(txt_file, ball_class_id)
            total_kept += kept
            total_removed += removed
            files_processed += 1

    # Обновляем data.yaml
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    data["nc"] = 1
    data["names"] = ["ball"]
    with open(data_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n{'='*50}")
    print(f"Очистка завершена!")
    print(f"  Файлов обработано      : {files_processed}")
    print(f"  Меток мяча оставлено   : {total_kept}")
    print(f"  Меток других классов   : {total_removed} (удалено)")
    print(f"  data.yaml обновлён: nc=1, names=['ball']")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оставляет только метки 'ball' в датасете")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ball_class_id", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Папка не найдена: {dataset_dir}")
    else:
        print(f"\nАнализируем датасет: {dataset_dir.name}")
        clean_dataset(dataset_dir, args.ball_class_id)
