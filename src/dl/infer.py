from pathlib import Path
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.infer.torch_model import Torch_model
from src.ptypes import img_size, label_to_name_mapping


def run_prod_infer(model_path: Path, path_to_data: Path, img_draft: bool) -> List[int]:
    preds = {}
    img_paths = [x for x in Path(path_to_data).glob("*.jpg")]

    model = Torch_model(model_path / "model.pt", label_to_name_mapping, *img_size)

    output_path = Path("output") / "results" / path_to_data.name
    output_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        if img_draft:
            img.draft("RGB", img_size)
        res = model(img)
        preds[img_path.name] = res[0]

    return preds


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    preds = run_prod_infer(
        model_path=Path(cfg.export.model_path),
        path_to_data=Path(cfg.export.path_to_data),
        img_draft=cfg.train.img_draft,
    )

    preds = pd.DataFrame(list(preds.items()), columns=["Id", "Expected"])
    preds.to_csv("subm.csv", index=False)
    print(preds)


if __name__ == "__main__":
    main()
