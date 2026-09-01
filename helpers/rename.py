import os
import pathlib
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
imgpath = os.path.join(parentdir, "dataset")
for curr_root,dirs,items in os.walk(imgpath):
    if items!=[]:
        for item in items:
            old_path = pathlib.Path(curr_root+os.sep+item)
            if old_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue
            temp_path = old_path.with_name(f'__temp__{item}')
            old_path.rename(temp_path)
        items=os.listdir(curr_root)
        parts = os.path.normpath(curr_root).split(os.sep)
        media=parts[-1]
        i=1
        for item in items:
            old_path = pathlib.Path(curr_root+os.sep+item)
            if old_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue
            new_path = old_path.with_name(f'{media}_{i}{old_path.suffix}')
            i+=1
            old_path.rename(new_path)