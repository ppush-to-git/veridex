import json
import cv2
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SIDTD_ROOT = Path(r"D:\veridex\dataset\SIDTD")

FAKE_JSON_DIR = SIDTD_ROOT / "Annotations" / "fakes"
FAKE_IMAGE_DIR = SIDTD_ROOT / "Images" / "fakes"
REAL_ANNOT_DIR = SIDTD_ROOT / "Annotations" / "reals"

OUTPUT_DIR = SIDTD_ROOT / "yolo_annotations"
VIS_DIR = SIDTD_ROOT / "annotated_fakes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND FAKE IMAGE
# ============================================================

def find_fake_image(fake_name):
    """
    Finds the fake image recursively.

    Handles:
        alb_id_00_fake_6_25
        alb_id_00_fake_6_25.jpg
    """

    candidates = [
        FAKE_IMAGE_DIR / fake_name,
        FAKE_IMAGE_DIR / f"{fake_name}.jpg",
        FAKE_IMAGE_DIR / f"{fake_name}.jpeg",
        FAKE_IMAGE_DIR / f"{fake_name}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Recursive fallback
    for path in FAKE_IMAGE_DIR.rglob("*"):
        if path.is_file() and path.stem == fake_name:
            return path

    return None


# ============================================================
# EXTRACT TEMPLATE + SOURCE IMAGE
# ============================================================

def get_template_and_source(src):
    """
    Example:

        alb_id_00.jpg

    becomes:

        template     = alb_id
        source_image = 00.jpg
    """

    src_path = Path(src)
    stem = src_path.stem

    parts = stem.split("_")

    if len(parts) < 2:
        return None, None

    # Last part is image number
    image_number = parts[-1]

    # Everything before image number is template
    template = "_".join(parts[:-1])

    source_image = image_number + src_path.suffix

    return template, source_image


# ============================================================
# LOAD REAL VIA ANNOTATION
# ============================================================

def load_real_annotation(template):

    annotation_path = REAL_ANNOT_DIR / f"{template}.json"

    if not annotation_path.exists():
        return None

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"[ERROR] Could not read {annotation_path}: {e}")
        return None


# ============================================================
# FIND SOURCE IMAGE INSIDE VIA JSON
# ============================================================

def find_source_image_annotation(real_data, source_image):

    metadata = real_data.get("_via_img_metadata", {})

    for image_data in metadata.values():

        filename = image_data.get("filename", "")

        if filename == source_image:
            return image_data

    return None


# ============================================================
# FIND FIELD BOUNDING BOX
# ============================================================

def find_field_bbox(image_data, field_name):

    regions = image_data.get("regions", [])

    for region in regions:

        region_attributes = region.get(
            "region_attributes",
            {}
        )

        shape_attributes = region.get(
            "shape_attributes",
            {}
        )

        annotated_field = region_attributes.get(
            "field_name"
        )

        if annotated_field == field_name:

            if shape_attributes.get("name") != "rect":
                continue

            x = shape_attributes.get("x", 0)
            y = shape_attributes.get("y", 0)
            width = shape_attributes.get("width", 0)
            height = shape_attributes.get("height", 0)

            return (
                int(x),
                int(y),
                int(width),
                int(height)
            )

    return None


# ============================================================
# BUILD ALL YOLO CLASSES AUTOMATICALLY
# ============================================================

def build_class_mapping():

    fields = set()

    print("\nScanning fake annotations for field names...")

    for json_file in FAKE_JSON_DIR.rglob("*.json"):

        try:

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            field = data.get("field")

            if field:
                field = field.strip()

                if field:
                    fields.add(field)

        except Exception as e:
            print(
                f"[WARNING] Could not read "
                f"{json_file.name}: {e}"
            )

    # Sort so class IDs are deterministic
    sorted_fields = sorted(fields)

    class_names = {
        field: class_id
        for class_id, field in enumerate(sorted_fields)
    }

    # --------------------------------------------------------
    # Save classes.txt
    # --------------------------------------------------------

    classes_txt = SIDTD_ROOT / "classes.txt"

    with open(classes_txt, "w", encoding="utf-8") as f:

        for field in sorted_fields:
            f.write(field + "\n")

    # --------------------------------------------------------
    # Save classes.json
    # --------------------------------------------------------

    classes_json = SIDTD_ROOT / "classes.json"

    with open(
        classes_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_names,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Print classes
    # --------------------------------------------------------

    print(f"\nFound {len(sorted_fields)} unique tampered fields.")

    print("\nYOLO classes:")

    for field, class_id in class_names.items():
        print(f"  {class_id}: {field}")

    print(f"\n[SAVED] {classes_txt}")
    print(f"[SAVED] {classes_json}")

    return class_names


# ============================================================
# PROCESS ONE FAKE JSON
# ============================================================

def process_fake_json(json_path, class_names):

    # --------------------------------------------------------
    # Read fake annotation
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        fake_data = json.load(f)

    fake_name = fake_data.get("name")
    src = fake_data.get("src")
    field_name = fake_data.get("field")

    if not fake_name:
        print(
            f"[ERROR] Missing 'name': "
            f"{json_path.name}"
        )
        return

    if not src:
        print(
            f"[ERROR] Missing 'src': "
            f"{json_path.name}"
        )
        return

    if not field_name:
        print(
            f"[ERROR] Missing 'field': "
            f"{json_path.name}"
        )
        return

    field_name = field_name.strip()

    print("\n" + "=" * 70)

    print(f"[PROCESS] {json_path.name}")

    print(f"  Fake name : {fake_name}")
    print(f"  Source    : {src}")
    print(f"  Field     : {field_name}")

    # --------------------------------------------------------
    # Find fake image
    # --------------------------------------------------------

    fake_image_path = find_fake_image(fake_name)

    if fake_image_path is None:

        print(
            f"[ERROR] Fake image not found: "
            f"{fake_name}"
        )

        return

    print(f"  Fake image: {fake_image_path}")

    # --------------------------------------------------------
    # Extract template and source image
    # --------------------------------------------------------

    template, source_image = get_template_and_source(src)

    if template is None:

        print(
            f"[ERROR] Could not parse source: "
            f"{src}"
        )

        return

    print(f"  Template  : {template}")
    print(f"  Real image: {source_image}")

    # --------------------------------------------------------
    # Load real annotation
    # --------------------------------------------------------

    real_data = load_real_annotation(template)

    if real_data is None:

        print(
            f"[ERROR] Real annotation not found: "
            f"{template}.json"
        )

        return

    # --------------------------------------------------------
    # Find source image in VIA annotation
    # --------------------------------------------------------

    image_data = find_source_image_annotation(
        real_data,
        source_image
    )

    if image_data is None:

        print(
            f"[ERROR] Source image "
            f"{source_image} not found inside "
            f"{template}.json"
        )

        return

    # --------------------------------------------------------
    # Find field bbox
    # --------------------------------------------------------

    bbox = find_field_bbox(
        image_data,
        field_name
    )

    if bbox is None:

        print(
            f"[ERROR] Field '{field_name}' "
            f"not found in {source_image}"
        )

        return

    x, y, width, height = bbox

    print(
        f"  BBox      : "
        f"x={x}, y={y}, "
        f"w={width}, h={height}"
    )

    # --------------------------------------------------------
    # Load fake image
    # --------------------------------------------------------

    image = cv2.imread(
        str(fake_image_path)
    )

    if image is None:

        print(
            f"[ERROR] Could not read fake image:"
            f" {fake_image_path}"
        )

        return

    img_height, img_width = image.shape[:2]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Real annotation coordinates may differ from fake image
    # dimensions.
    #
    # Scale bbox if dimensions differ.
    # --------------------------------------------------------

    # We currently assume the original annotation coordinates
    # match the source image dimensions.
    #
    # Since the SIDTD fake is generated from the source image,
    # this should normally be correct.
    #
    # Still clamp everything to the actual fake image.

    x1 = max(0, min(x, img_width - 1))
    y1 = max(0, min(y, img_height - 1))

    x2 = max(
        x1 + 1,
        min(x + width, img_width - 1)
    )

    y2 = max(
        y1 + 1,
        min(y + height, img_height - 1)
    )

    # --------------------------------------------------------
    # DRAW VISUALIZATION
    # --------------------------------------------------------

    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    label = f"TAMPER: {field_name}"

    # Prevent text from going outside the image
    text_y = max(30, y1 - 10)

    cv2.putText(
        annotated,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    visual_output = (
        VIS_DIR /
        f"{fake_name}_annotated.jpg"
    )

    cv2.imwrite(
        str(visual_output),
        annotated
    )

    # --------------------------------------------------------
    # YOLO CLASS
    # --------------------------------------------------------

    if field_name not in class_names:

        print(
            f"[ERROR] Field '{field_name}' "
            f"not present in generated class mapping."
        )

        return

    class_id = class_names[field_name]

    # --------------------------------------------------------
    # CONVERT BBOX TO YOLO FORMAT
    #
    # YOLO:
    #
    # class_id center_x center_y width height
    #
    # All coordinates normalized 0 -> 1
    # --------------------------------------------------------

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    center_x = x1 + bbox_width / 2
    center_y = y1 + bbox_height / 2

    center_x /= img_width
    center_y /= img_height

    normalized_width = (
        bbox_width / img_width
    )

    normalized_height = (
        bbox_height / img_height
    )

    # Clamp normalized values
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    normalized_width = max(
        0.0,
        min(1.0, normalized_width)
    )
    normalized_height = max(
        0.0,
        min(1.0, normalized_height)
    )

    # --------------------------------------------------------
    # SAVE YOLO TXT
    # --------------------------------------------------------

    yolo_output = (
        OUTPUT_DIR /
        f"{fake_name}.txt"
    )

    with open(
        yolo_output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}\n"
        )

    print(
        f"[SAVED] Visualization: "
        f"{visual_output}"
    )

    print(
        f"[SAVED] YOLO label: "
        f"{yolo_output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIDTD FAKE FIELD -> YOLO ANNOTATION GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not FAKE_JSON_DIR.exists():

        print(
            f"[ERROR] Fake annotation directory "
            f"does not exist:\n{FAKE_JSON_DIR}"
        )

        return

    if not FAKE_IMAGE_DIR.exists():

        print(
            f"[ERROR] Fake image directory "
            f"does not exist:\n{FAKE_IMAGE_DIR}"
        )

        return

    if not REAL_ANNOT_DIR.exists():

        print(
            f"[ERROR] Real annotation directory "
            f"does not exist:\n{REAL_ANNOT_DIR}"
        )

        return

    # --------------------------------------------------------
    # Find fake JSON files
    # --------------------------------------------------------

    json_files = list(
        FAKE_JSON_DIR.rglob("*.json")
    )

    print(
        f"\nFound {len(json_files)} "
        f"fake annotation files."
    )

    if not json_files:

        print("[ERROR] No fake JSON files found.")

        return

    # --------------------------------------------------------
    # Automatically generate YOLO classes
    # --------------------------------------------------------

    class_names = build_class_mapping()

    # --------------------------------------------------------
    # Process every fake annotation
    # --------------------------------------------------------

    success = 0
    failed = 0

    for json_file in json_files:

        try:

            process_fake_json(
                json_file,
                class_names
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {json_file.name}: "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Processed : {len(json_files)}")
    print(f"Completed : {success}")
    print(f"Errors    : {failed}")

    print(
        f"\nYOLO labels:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nVisualized:"
        f"\n{VIS_DIR}"
    )

    print(
        f"\nClasses:"
        f"\n{SIDTD_ROOT / 'classes.txt'}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()