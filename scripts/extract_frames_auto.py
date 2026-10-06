"""
Скрипт нарезки видео на кадры для датасета YOLO.

Обходит все папки extracted_frames/<table>/source_<table>/ и
нарезает видео, сохраняя кадры в with_ball/.

Использование:
    # Обычный запуск (все столы, 2 кадра в секунду):
    python scripts/extract_frames_auto.py

    # Тест — только 30 кадров с каждого видео:
    python scripts/extract_frames_auto.py --test

    # Указать другую папку:
    python scripts/extract_frames_auto.py --base_dir data/ML_workflow/extracted_frames/bonzini
"""

import cv2
import argparse
from pathlib import Path


def process_all_videos(base_dir: Path, test_mode: bool = False, fps_target: float = 2.0):
    """
    Обходит все подпапки base_dir, находит видеофайлы в source_* папках
    и нарезает кадры в соседнюю папку with_ball/.
    """
    extensions = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = [p for p in base_dir.rglob("*") if p.suffix.lower() in extensions]

    if not video_files:
        print(f"\nВидео не найдены в: {base_dir}")
        print("Убедись, что .mov/.mp4 файлы лежат в папках source_*/")
        return

    max_per_video = 30 if test_mode else None
    mode_label = "ТЕСТ (30 кадров)" if test_mode else "ПОЛНЫЙ РЕЖИМ"
    print(f"\n{'='*55}")
    print(f"  Режим: {mode_label}")
    print(f"  Найдено видео: {len(video_files)}")
    print(f"  Папка: {base_dir}")
    print(f"{'='*55}")

    total_saved = 0

    for video_path in sorted(video_files):
        # Находим папку стола (родитель source_* папки)
        table_dir = video_path.parent.parent
        table_name = table_dir.name
        out_dir = table_dir / "with_ball"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{table_name.upper()}] {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Ошибка: не удалось открыть файл")
            continue

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / source_fps

        # Берём каждый N-й кадр, чтобы получить fps_target кадров в секунду
        skip = max(1, int(source_fps / fps_target))

        print(f"  Длина: {duration_sec:.0f} сек | FPS: {source_fps:.0f} | каждый {skip}-й кадр")

        count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % skip == 0:
                out_name = out_dir / f"{video_path.stem}_{saved:04d}.jpg"
                cv2.imwrite(str(out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

                if max_per_video and saved >= max_per_video:
                    print(f"  Тест: достигнут лимит {max_per_video} кадров, останавливаем")
                    break

            count += 1

        cap.release()
        total_saved += saved
        print(f"  Сохранено: {saved} кадров -> {out_dir}")

    print(f"\n{'='*55}")
    print(f"  ИТОГО сохранено: {total_saved} кадров")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Нарезка видео на кадры для YOLO датасета")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/ML_workflow/extracted_frames",
        help="Путь к папке extracted_frames"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим: сохранить только 30 кадров с каждого видео"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Кол-во кадров в секунду для сохранения (по умолчанию: 2)"
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Папка не найдена: {base_path}")
    else:
        process_all_videos(base_path, test_mode=args.test, fps_target=args.fps)
