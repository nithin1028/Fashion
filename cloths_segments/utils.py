from pathlib import Path
from typing  import Union,Dict,List,Tuple
def get_id_file_paths(path:Union[str,Path])->Dict[str,Path]:
    return {x.stem:x for x in Path(path).glob("*.*")}
def get_samples(images_path:Path,mask_path:Path)->List[Tuple[Path,Path]]:
    image2path=get_id_file_paths(images_path)
    mask2path=get_id_file_paths(mask_path)
    missing=[file_id for file_id in image2path if file_id not in mask2path]
    if missing:
        print(f"[get_samples] skipping {len(missing)} images without a mask, e.g. {missing[:3]}")
    samples=[(image2path[file_id],mask2path[file_id]) for file_id in image2path if file_id in mask2path]
    if not samples:
        raise RuntimeError(f"no image/mask pairs found: {images_path} vs {mask_path}")
    return samples
