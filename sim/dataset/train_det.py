"""Entrena YOLOv8n detector local CPU."""
from pathlib import Path
import time
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "det" / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "detect"

def main():
    print(f"[train_det] data_yaml={DATA_YAML}")
    print(f"[train_det] exists={DATA_YAML.exists()}")
    if DATA_YAML.exists():
        print(DATA_YAML.read_text(encoding="utf-8"))
    import glob
    train_imgs = glob.glob(str(REPO_ROOT / "data" / "det" / "train" / "images" / "*.*"))
    val_imgs = glob.glob(str(REPO_ROOT / "data" / "det" / "val" / "images" / "*.*"))
    print(f"train images flat: {len(train_imgs)} val: {len(val_imgs)}")
    model = YOLO("yolov8n.pt")
    start = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=320,
        batch=8,
        device="cpu",
        workers=2,
        patience=3,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=False,
        verbose=True,
        save=True,
        plots=True,
        val=True,
    )
    elapsed = time.time() - start
    print(f"[train_det] DONE elapsed={elapsed/60:.1f} min")
    try:
        (REPO_ROOT / "runs" / "detect" / "train" / "train_time.txt").write_text(f"{elapsed}\n", encoding="utf-8")
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
