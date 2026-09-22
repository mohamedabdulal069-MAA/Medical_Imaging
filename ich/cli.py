import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Audited ICH research baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--dicoms", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--baseline", choices=["vgg16", "resnet50", "densenet121", "efficientnetb0", "inceptionv3", "ensemble", "pca_ensemble"], default="resnet50")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--components", type=int, default=512)
    train.add_argument("--weighted-loss", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--partition", choices=["test", "validation"], default="test")
    predict = sub.add_parser("predict")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--dicom", required=True)
    explain = sub.add_parser("explain")
    explain.add_argument("--bundle", required=True)
    explain.add_argument("--dicom", required=True)
    explain.add_argument("--layer", required=True)
    explain.add_argument("--label", required=True, type=int, choices=range(6))
    explain.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .dataset import build_manifest, split_manifest
        destination = Path(args["output"])
        if destination.exists():
            raise FileExistsError("Preserve existing split manifests")
        manifest = split_manifest(build_manifest(args["labels"], args["dicoms"]), seed=args["seed"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(destination, index=False)
        print(manifest.groupby("partition").agg(images=("image_id", "size"), patients=("patient_id", "nunique")))
    elif command in ("train", "evaluate"):
        from .dataset import load_manifest
        args["manifest"] = load_manifest(args["manifest"])
        if command == "train":
            from .training import train
            result = train(**args)
        else:
            from .evaluation import evaluate
            result = evaluate(**args)
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        from .evaluation import Predictor
        predictor = Predictor(args["bundle"])
        if command == "predict":
            print(json.dumps(predictor.predict_path(args["dicom"]), indent=2))
        else:
            import numpy as np
            from .preprocessing import load_image
            from .explainability import gradcam
            if predictor.pca is not None:
                raise ValueError("PCA attribution is not implemented; no surrogate CNN explanation is substituted")
            image = load_image(args["dicom"], predictor.metadata["size"])[None]
            heatmap = gradcam(predictor.model, image, args["layer"], args["label"])
            with open(args["output"], "xb") as stream:
                np.save(stream, heatmap, allow_pickle=False)


if __name__ == "__main__":
    main()
