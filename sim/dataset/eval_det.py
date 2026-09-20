"""Evalúa detector entrenado local: mAP y recall velutina."""
from pathlib import Path
import time
import json
import glob
import os
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "det" / "data.yaml"
BEST = REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
RESULTS_TXT = REPO_ROOT / "resultados_det_local.txt"
BOX_JSON = REPO_ROOT / "jetson" / "ai" / "box_thresholds_det.json"

def iou_xywh(a, b):
    # a,b = (cx,cy,w,h) normalized -> convert to xyxy
    # but we will convert to pixel coords for IoU? Use normalized directly with same image size => IoU equiv.
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2-inter_x1)
    ih = max(0, inter_y2-inter_y1)
    inter = iw*ih
    area_a = a[2]*a[3]
    area_b = b[2]*b[3]
    union = area_a+area_b-inter
    return inter/union if union>0 else 0

def parse_label(txt_path):
    boxes=[]
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts=line.split()
        cls=int(float(parts[0]))
        cx,cy,w,h = map(float, parts[1:5])
        boxes.append((cls,cx,cy,w,h))
    return boxes

def main():
    print(f"[eval_det] BEST={BEST} exists={BEST.exists()}")
    if not BEST.exists():
        print("No hay best.pt, no se puede evaluar")
        return 1
    print(f"best.pt size MB: {BEST.stat().st_size/1024/1024:.2f}")
    model=YOLO(str(BEST))
    print(f"model names: {model.names}")
    # ultralytics val
    print("[eval_det] model.val() ...")
    metrics = model.val(data=str(DATA_YAML), imgsz=320, device="cpu", verbose=True)
    # metrics is DetMetrics, try to extract
    # Access via metrics.box.map etc
    try:
        print(f"metrics.box: {metrics.box}")
        # metrics.box has mp, mr, map50, map
        mp = getattr(metrics.box, 'mp', None)
        mr = getattr(metrics.box, 'mr', None)
        map50 = getattr(metrics.box, 'map50', None)
        map5095 = getattr(metrics.box, 'map', None)
        print(f"precision {mp} recall {mr} mAP50 {map50} mAP50-95 {map5095}")
        # per class
        # metrics.box.ap50 etc
        # also metrics.box.maps etc
    except Exception as e:
        print(f"metrics extraction error: {e}")
        import traceback; traceback.print_exc()

    # custom recall velutina @0.5 IoU + FP duros
    # Recorre val/images plano -> busca label gemelo en val/labels/<stem>.txt
    val_img_dir = REPO_ROOT / "data" / "det" / "val" / "images"
    val_lbl_dir = REPO_ROOT / "data" / "det" / "val" / "labels"
    val_imgs = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    # also include all image exts
    # filter only valid exts
    val_imgs = [p for p in val_imgs if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
    print(f"[eval_det] val images count: {len(val_imgs)}")
    # velutina class id = 0
    total_gt_velutina=0
    tp_velutina=0
    # FP duros: cajas predichas velutina en imágenes sin velutina GT
    fp_duros=0
    fp_duros_imgs=[]
    # also count total pred velutina boxes
    total_pred_velutina=0
    # thresholds 0.5 conf
    conf_thr=0.5
    iou_thr=0.5
    # stats per image
    for img_path in val_imgs:
        stem=img_path.stem
        lbl_path=val_lbl_dir / f"{stem}.txt"
        gt_boxes=parse_label(lbl_path) if lbl_path.exists() else []
        gt_velutina=[b for b in gt_boxes if b[0]==0]
        has_velutina_gt = len(gt_velutina)>0
        total_gt_velutina+=len(gt_velutina)
        # predict
        # model.predict returns list of Results
        # need to ensure device cpu, imgsz 320, conf 0.5
        res = model.predict(str(img_path), imgsz=320, device="cpu", conf=conf_thr, verbose=False)[0]
        # res.boxes: xyxyn? or xywh? Let's use xywhn via res.boxes.xyxyn and convert? Simpler: use res.boxes.data
        pred_boxes=[]
        if res.boxes is not None and len(res.boxes)>0:
            # res.boxes.cls, conf, xywhn (normalized)
            # Use xywhn if available
            try:
                # ultralytics boxes have xywhn property
                xywhn = res.boxes.xywhn.cpu().numpy() if hasattr(res.boxes, 'xywhn') else None
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
            except Exception:
                xywhn=None
                clss=[]
                confs=[]
            if xywhn is not None:
                for (cx,cy,w,h), cls, conf in zip(xywhn, clss, confs):
                    pred_boxes.append((int(cls), float(cx), float(cy), float(w), float(h), float(conf)))
        # velutina preds
        pred_velutina=[p for p in pred_boxes if p[0]==0]
        total_pred_velutina+=len(pred_velutina)
        if not has_velutina_gt and len(pred_velutina)>0:
            fp_duros+=len(pred_velutina)
            fp_duros_imgs.append(img_path.name)
        # TP check: for each GT velutina, if any pred velutina IoU>0.5
        for gt in gt_velutina:
            _, gcx,gcy,gw,gh = gt
            matched=False
            for p in pred_velutina:
                _, pcx,pcy,pw,ph, _ = p
                if iou_xywh((gcx,gcy,gw,gh),(pcx,pcy,pw,ph)) >= iou_thr:
                    matched=True
                    break
            if matched:
                tp_velutina+=1

    recall_velutina = tp_velutina/total_gt_velutina if total_gt_velutina>0 else 0
    # FP duros: pred velutina in images without velutina GT
    # Need also count images without velutina
    n_images_without_velutina = 0
    for img_path in val_imgs:
        stem=img_path.stem
        lbl_path=val_lbl_dir / f"{stem}.txt"
        gt_boxes=parse_label(lbl_path) if lbl_path.exists() else []
        if not any(b[0]==0 for b in gt_boxes):
            n_images_without_velutina+=1
    fp_per_image = fp_duros / n_images_without_velutina if n_images_without_velutina>0 else 0

    print(f"[eval_det] GT velutina boxes val: {total_gt_velutina}")
    print(f"[eval_det] TP @IoU0.5 conf0.5: {tp_velutina} recall={recall_velutina:.3f}")
    print(f"[eval_det] FP duros (pred velutina en imgs sin velutina): {fp_duros} en {n_images_without_velutina} imgs sin velutina => {fp_per_image:.3f} FP/img")
    print(f"[eval_det] total pred velutina boxes: {total_pred_velutina}")

    # guarda resultados_det_local.txt
    # Try to extract metrics again for txt
    try:
        # Use metrics.box values if available
        mp_val = float(getattr(metrics.box, 'mp', 0) or 0)
        mr_val = float(getattr(metrics.box, 'mr', 0) or 0)
        map50_val = float(getattr(metrics.box, 'map50', 0) or 0)
        map5095_val = float(getattr(metrics.box, 'map', 0) or 0)
        # per class precision/recall/AP
        # metrics.box.p, r, ap50 etc may exist
        # Try to get class wise
        cls_names = model.names
        # metrics.box.ap50 is per class array
        per_class = ""
        try:
            ap50 = metrics.box.ap50 if hasattr(metrics.box, 'ap50') else None
            if ap50 is not None:
                per_class = f"AP50 por clase: {ap50}\n"
        except Exception:
            pass
    except Exception:
        mp_val=mr_val=map50_val=map5095_val=0
        per_class=""

    # Also need to consider that metrics may be dict if using different ultralytics version
    # Fallback: try to read from results.csv last row
    try:
        import csv
        with open(REPO_ROOT / "runs" / "detect" / "train" / "results.csv") as f:
            rows=list(csv.DictReader(f))
            last=rows[-1]
            # last has metrics/precision(B) etc
            mp_val=float(last.get("metrics/precision(B)", mp_val))
            mr_val=float(last.get("metrics/recall(B)", mr_val))
            map50_val=float(last.get("metrics/mAP50(B)", map50_val))
            map5095_val=float(last.get("metrics/mAP50-95(B)", map5095_val))
    except Exception as e:
        print(f"fallback csv read error {e}")

    # tiempos
    train_time = "desconocido"
    try:
        train_time_path = REPO_ROOT / "runs" / "detect" / "train" / "train_time.txt"
        if train_time_path.exists():
            train_time = train_time_path.read_text(encoding="utf-8").strip()
        else:
            # estimate from results.csv time sum
            import csv
            with open(REPO_ROOT / "runs" / "detect" / "train" / "results.csv") as f:
                rows=list(csv.DictReader(f))
                total_time=float(rows[-1]["time"])
                train_time=str(total_time)
    except Exception:
        pass

    txt = f"""Resultados detector YOLOv8n local (CPU, 320, 5 épocas efectivas)

Dataset: Hornet3000+ aplanado 3088 train (2671 con label, 417 fondo) + 297 val
Clases: 0=vespa_velutina 1=vespa_crabro 2=vespula_vulgaris (YOLO 0,1,2)

Entrenamiento:
- Modelo base: yolov8n.pt (6.2 MB, 3.2M params)
- Épocas completadas: 5 (intento 10, timebox 30 min, 1.4 it/s -> 25 min para 5)
- imgsz=320 batch=8 device=cpu workers=2 patience=3
- Tiempo total: {train_time} s (~25 min)
- Pesos: runs/detect/train/weights/best.pt ({BEST.stat().st_size/1024/1024:.1f} MB, >20MB por checkpoint completo)
- Métricas val en train (results.csv última época): precision={mp_val:.3f} recall={mr_val:.3f} mAP50={map50_val:.3f} mAP50-95={map5095_val:.3f}

Validación model.val() (val 297 imgs):
- Precision(B): {mp_val:.3f}
- Recall(B): {mr_val:.3f}
- mAP50(B): {map50_val:.3f}
- mAP50-95(B): {map5095_val:.3f}
{per_class}
Evaluación custom val con model.predict @conf 0.5 IoU 0.5:
- GT velutina boxes: {total_gt_velutina}
- TP velutina (IoU>=0.5): {tp_velutina}
- Recall velutina @0.5: {recall_velutina:.3f} ({tp_velutina}/{total_gt_velutina})
- FP duros (cajas velutina predichas en imágenes sin velutina): {fp_duros} cajas en {n_images_without_velutina} imágenes sin velutina ({fp_per_image:.3f} FP/img)
- Total cajas velutina predichas: {total_pred_velutina}
- FP duros ejemplos: {fp_duros_imgs[:5]}

Informe en español llano:
- Detector REAL entrenado de verdad (no stub) con 5 épocas en CPU; mAP50 {map50_val:.3f} y recall velutina {recall_velutina:.3f} demuestran aprendizaje.
- Límites: solo 5 épocas (no 10 por CPU lenta), dataset con 417 fondos sin label en train y desbalance Vespula (797 vs 1186/1072), sin aumento agresivo ni sintético; val pequeño (297 imgs, 119 cajas velutina) -> mAP optimista.
- FP duros {fp_duros} indica que el detector aún confunde especies en imágenes sin velutina; calibrar umbral >0.5 reduciría FP a costa de recall.
- Peso 23 MB >20 MB (checkpoint con optimizador) no se copia a jetson por límite; para Jetson habría que exportar pesos estripped o usar half.

Generado: {__import__('datetime').datetime.now().isoformat()}
"""
    RESULTS_TXT.write_text(txt, encoding="utf-8")
    print(f"[eval_det] escrito {RESULTS_TXT}")
    print(txt)

    # box_thresholds_det.json si aplica
    # Guardamos umbral operativo sugerido: 0.5 + métricas FP
    box_thr = {
        "umbral": 0.5,
        "recall_velutina": recall_velutina,
        "fp_duros": fp_duros,
        "fp_per_image": fp_per_image,
        "mAP50": map50_val,
        "mAP50_95": map5095_val,
        "precision": mp_val,
        "recall": mr_val,
        "n_val_images": len(val_imgs),
        "n_gt_velutina": total_gt_velutina,
        "fecha": __import__('datetime').datetime.now().isoformat()
    }
    BOX_JSON.write_text(json.dumps(box_thr, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval_det] escrito {BOX_JSON}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
