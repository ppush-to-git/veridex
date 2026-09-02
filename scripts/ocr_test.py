#!/usr/bin/env python3
"""
Document OCR and Information Extraction Program
------------------------------------------------
Extracts, structures, and visualizes text and key identity fields from
photos and scanned images across SIDTD, MIDV_2020, FCD-V, and custom documents.
"""

import os
import sys
import re
import math
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Suppress PyTorch quantization / pin_memory deprecation notices for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure proper stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import easyocr
from PIL import Image, ImageDraw, ImageFont


class DocumentOCR:
    """
    Production-grade Document OCR engine with spatial key-value pairing,
    entity parsing, and visual dashboard generation.
    """

    def __init__(self, languages: List[str] = None, gpu: Optional[bool] = None):
        """
        Initialize the OCR reader.
        Default languages: English ('en') and Albanian/Latin ('sq') for broad ID coverage.
        """
        if languages is None:
            languages = ['en', 'sq']

        if gpu is None:
            gpu = torch.cuda.is_available()

        self.gpu = gpu
        self.languages = languages
        self.reader = easyocr.Reader(languages, gpu=self.gpu, verbose=False)

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast and sharpness for optimal character recognition.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def extract_raw_ocr(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run EasyOCR and normalize bounding boxes into standard structured dicts.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        results = self.reader.readtext(str(image_path))
        items = []

        for bbox, text, conf in results:
            text = text.strip()
            if not text:
                continue

            pts = np.array(bbox, dtype=np.int32)
            x_min = int(np.min(pts[:, 0]))
            x_max = int(np.max(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            y_max = int(np.max(pts[:, 1]))
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            height = max(1, y_max - y_min)
            width = max(1, x_max - x_min)

            items.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [[int(p[0]), int(p[1])] for p in pts],
                "rect": [x_min, y_min, x_max, y_max],
                "center": (center_x, center_y),
                "height": height,
                "width": width,
            })

        return items

    def _find_value_below_label(
        self,
        label_item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        max_y_dist_ratio: float = 3.5,
        max_x_offset_ratio: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find candidate value text block directly below a given field label block.
        """
        lx1, ly1, lx2, ly2 = label_item["rect"]
        l_height = label_item["height"]
        l_width = label_item["width"]
        l_cx, l_cy = label_item["center"]

        best_candidate = None
        min_dist = float("inf")

        for item in all_items:
            if item == label_item:
                continue
            ix1, iy1, ix2, iy2 = item["rect"]
            i_cx, i_cy = item["center"]

            # Must be positioned below the label top
            if iy1 < ly1:
                continue

            y_dist = iy1 - ly2
            if y_dist < -l_height * 0.4:
                continue
            if y_dist > l_height * max_y_dist_ratio:
                continue

            overlap = max(0, min(lx2, ix2) - max(lx1, ix1))
            x_dist = abs(i_cx - l_cx)

            if overlap > 0 or x_dist < l_width * max_x_offset_ratio:
                dist_score = y_dist + x_dist * 0.3
                if dist_score < min_dist:
                    min_dist = dist_score
                    best_candidate = item

        return best_candidate

    def _find_value_right_of_label(
        self,
        label_item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        max_x_dist_ratio: float = 3.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find candidate value text block immediately to the right of a label on the same line.
        """
        lx1, ly1, lx2, ly2 = label_item["rect"]
        l_height = label_item["height"]
        l_cy = label_item["center"][1]

        best_candidate = None
        min_dist = float("inf")

        for item in all_items:
            if item == label_item:
                continue
            ix1, iy1, ix2, iy2 = item["rect"]
            i_cy = item["center"][1]

            if abs(i_cy - l_cy) > l_height * 0.8:
                continue

            x_dist = ix1 - lx2
            if -5 <= x_dist <= l_height * max_x_dist_ratio * 4:
                if x_dist < min_dist:
                    min_dist = x_dist
                    best_candidate = item

        return best_candidate

    def parse_document_fields(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse and structure raw OCR items into semantic identity fields.
        """
        fields: Dict[str, Dict[str, Any]] = {
            "document_title": {"value": None, "confidence": 0.0, "bbox": None},
            "document_type": {"value": None, "confidence": 0.0, "bbox": None},
            "surname": {"value": None, "confidence": 0.0, "bbox": None},
            "given_names": {"value": None, "confidence": 0.0, "bbox": None},
            "full_name": {"value": None, "confidence": 0.0, "bbox": None},
            "nationality": {"value": None, "confidence": 0.0, "bbox": None},
            "document_number": {"value": None, "confidence": 0.0, "bbox": None},
            "personal_number": {"value": None, "confidence": 0.0, "bbox": None},
            "place_of_birth": {"value": None, "confidence": 0.0, "bbox": None},
            "date_of_birth": {"value": None, "confidence": 0.0, "bbox": None},
            "date_of_issue": {"value": None, "confidence": 0.0, "bbox": None},
            "date_of_expiry": {"value": None, "confidence": 0.0, "bbox": None},
            "sex": {"value": None, "confidence": 0.0, "bbox": None},
            "authority": {"value": None, "confidence": 0.0, "bbox": None},
            "email": {"value": None, "confidence": 0.0, "bbox": None},
            "mrz_lines": {"value": [], "confidence": 0.0, "bbox": None},
        }

        date_pattern = re.compile(r"\b(\d{2}[-/\.]\d{2}[-/\.]\d{4}|\d{4}[-/\.]\d{2}[-/\.]\d{2})\b")
        personal_no_pattern = re.compile(r"\b([A-Z]\d{8,9}[A-Z]|\d{9,10})\b")
        card_no_pattern = re.compile(r"\b(\d{8,10})\b")
        mrz_pattern = re.compile(r"[A-Z0-9<]{10,}<{2,}[A-Z0-9<]*")

        used_items = set()

        # Step 1: Detect Document Title / Header
        for idx, item in enumerate(items):
            text_upper = item["text"].upper()
            if any(h in text_upper for h in ["REPUBLIKA", "REPUBLIC", "LETERNJOFTIM", "CANADA", "PASSPORT", "IDENTITY CARD", "DRIVING LICENCE"]):
                if not fields["document_title"]["value"]:
                    clean_title = re.sub(r"[^A-Za-z0-9\s/ëËçÇ\-]", "", item["text"]).strip()
                    fields["document_title"] = {
                        "value": clean_title,
                        "confidence": item["confidence"],
                        "bbox": item["bbox"]
                    }
                    used_items.add(idx)

        # Step 2: Extract explicit Inline Key: Value patterns (e.g. FCD-V format "First Name: Ernest")
        for idx, item in enumerate(items):
            text = item["text"]
            if ":" in text:
                k, v = text.split(":", 1)
                k_norm = k.strip().lower()
                v_clean = v.strip()
                if not v_clean:
                    cand = self._find_value_right_of_label(item, items)
                    if cand:
                        v_clean = cand["text"].strip()

                if v_clean:
                    if "first" in k_norm:
                        fields["given_names"] = {"value": v_clean, "confidence": item["confidence"], "bbox": item["bbox"]}
                        used_items.add(idx)
                    elif "last" in k_norm or "surname" in k_norm:
                        fields["surname"] = {"value": v_clean, "confidence": item["confidence"], "bbox": item["bbox"]}
                        used_items.add(idx)
                    elif "gender" in k_norm or "sex" in k_norm:
                        fields["sex"] = {"value": v_clean, "confidence": item["confidence"], "bbox": item["bbox"]}
                        used_items.add(idx)
                    elif "email" in k_norm:
                        fields["email"] = {"value": v_clean, "confidence": item["confidence"], "bbox": item["bbox"]}
                        used_items.add(idx)
                    elif "id" in k_norm or "card" in k_norm or "no" in k_norm:
                        fields["document_number"] = {"value": v_clean, "confidence": item["confidence"], "bbox": item["bbox"]}
                        used_items.add(idx)

        # Step 3: Spatial Label-to-Value Parsing (e.g. SIDTD & MIDV stacked layouts)
        for idx, item in enumerate(items):
            t_low = item["text"].lower()

            # Surname
            if ("surname" in t_low or "mbiemri" in t_low) and not fields["surname"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    clean_v = re.sub(r"[^A-Za-z\-'\s]", "", val_cand["text"]).strip()
                    fields["surname"] = {"value": clean_v, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Given Name
            elif ("given name" in t_low or "emri" in t_low or "first name" in t_low) and not fields["given_names"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    clean_v = re.sub(r"[^A-Za-z\-'\s]", "", val_cand["text"]).strip()
                    fields["given_names"] = {"value": clean_v, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Nationality
            elif ("nationality" in t_low or "shtetësia" in t_low or "shtetesia" in t_low) and not fields["nationality"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    fields["nationality"] = {"value": val_cand["text"].strip(), "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Place of birth
            elif ("place of birth" in t_low or "vendlindja" in t_low or "lieu de naissance" in t_low) and not fields["place_of_birth"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    clean_v = val_cand["text"].strip().rstrip(";")
                    right_cand = self._find_value_right_of_label(val_cand, items)
                    if right_cand and len(right_cand["text"]) <= 4:
                        clean_v = f"{clean_v} {right_cand['text']}".strip()
                    fields["place_of_birth"] = {"value": clean_v, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Date of Birth
            elif ("date of birth" in t_low or "datëlindja" in t_low or "datelindja" in t_low or "birth" in t_low) and not fields["date_of_birth"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    d_match = date_pattern.search(val_cand["text"])
                    val = d_match.group(0) if d_match else val_cand["text"].strip()
                    fields["date_of_birth"] = {"value": val, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Date of Issue
            elif ("date of issue" in t_low or "lëshimit" in t_low or "leshimit" in t_low or "issue" in t_low) and not fields["date_of_issue"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    d_match = date_pattern.search(val_cand["text"])
                    val = d_match.group(0) if d_match else val_cand["text"].strip()
                    fields["date_of_issue"] = {"value": val, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Date of Expiry
            elif ("date of expiry" in t_low or "skadimit" in t_low or "expiry" in t_low or "expiration" in t_low) and not fields["date_of_expiry"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    d_match = date_pattern.search(val_cand["text"])
                    val = d_match.group(0) if d_match else val_cand["text"].strip()
                    fields["date_of_expiry"] = {"value": val, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Sex / Gender
            elif ("sex" in t_low or "gjinia" in t_low or "gender" in t_low) and not fields["sex"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    s_clean = val_cand["text"].strip().upper()
                    if s_clean in ["M", "F", "MALE", "FEMALE", "X"]:
                        fields["sex"] = {"value": s_clean, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                        used_items.add(idx)

            # Card / Document Number
            elif ("card no" in t_low or "letërnjoftim" in t_low or "document no" in t_low or "id no" in t_low) and not fields["document_number"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    c_match = card_no_pattern.search(val_cand["text"])
                    val = c_match.group(0) if c_match else val_cand["text"].strip()
                    fields["document_number"] = {"value": val, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Personal Number / National ID
            elif ("personal no" in t_low or "nr. personal" in t_low or "personal" in t_low) and not fields["personal_number"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    p_match = personal_no_pattern.search(val_cand["text"])
                    val = p_match.group(0) if p_match else val_cand["text"].strip()
                    fields["personal_number"] = {"value": val, "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

            # Authority
            elif ("authority" in t_low or "autoriteti" in t_low) and not fields["authority"]["value"]:
                val_cand = self._find_value_below_label(item, items)
                if val_cand:
                    fields["authority"] = {"value": val_cand["text"].strip(), "confidence": val_cand["confidence"], "bbox": val_cand["bbox"]}
                    used_items.add(idx)

        # Step 4: Search for MRZ lines and unassigned pattern matches
        mrz_list = []
        for idx, item in enumerate(items):
            text = item["text"]
            if mrz_pattern.search(text) or ("<" in text and len(text) > 12):
                mrz_list.append(text)
                used_items.add(idx)

        if mrz_list:
            fields["mrz_lines"] = {"value": mrz_list, "confidence": 0.95, "bbox": None}

        # Step 5: Fallback scan for remaining unmatched entities
        for idx, item in enumerate(items):
            if idx in used_items:
                continue
            text = item["text"]

            # Personal ID fallback (e.g. J11120296E)
            if not fields["personal_number"]["value"]:
                p_match = personal_no_pattern.search(text)
                if p_match and len(p_match.group(0)) >= 8:
                    fields["personal_number"] = {"value": p_match.group(0), "confidence": item["confidence"], "bbox": item["bbox"]}
                    continue

            # Card Number fallback (e.g. 367253746)
            if not fields["document_number"]["value"]:
                c_match = card_no_pattern.search(text)
                if c_match and len(c_match.group(0)) >= 8:
                    fields["document_number"] = {"value": c_match.group(0), "confidence": item["confidence"], "bbox": item["bbox"]}
                    continue

            # Standalone Sex indicator
            if not fields["sex"]["value"] and text.strip().upper() in ["M", "F"]:
                fields["sex"] = {"value": text.strip().upper(), "confidence": item["confidence"], "bbox": item["bbox"]}
                continue

        # Synthesize Full Name
        surname = fields["surname"]["value"] or ""
        given = fields["given_names"]["value"] or ""
        if surname or given:
            fields["full_name"]["value"] = f"{surname} {given}".strip()
            fields["full_name"]["confidence"] = min(
                fields["surname"]["confidence"] or 1.0,
                fields["given_names"]["confidence"] or 1.0
            )

        return fields

    def generate_visual_dashboard(
        self,
        image_path: Path,
        raw_items: List[Dict[str, Any]],
        parsed_fields: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Render a high-contrast, beautiful visual dashboard showing:
        - Left: Document image with highlighted bounding boxes and field badges.
        - Right: High-visibility structured information panel.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            raise ValueError(f"Failed to read image at {image_path}")

        h_orig, w_orig = orig_bgr.shape[:2]

        # Standardize left document width to 800px while maintaining aspect ratio
        target_doc_w = 800
        target_doc_h = int(h_orig * (target_doc_w / w_orig))
        resized_doc = cv2.resize(orig_bgr, (target_doc_w, target_doc_h), interpolation=cv2.INTER_LANCZOS4)

        scale_x = target_doc_w / w_orig
        scale_y = target_doc_h / h_orig

        doc_overlay = resized_doc.copy()

        # Build map of parsed field bboxes to highlight them distinctly
        highlight_boxes = {}
        for f_name, f_data in parsed_fields.items():
            if f_data.get("bbox") and f_data.get("value"):
                bbox_tuple = tuple(f_data["bbox"][0])
                highlight_boxes[bbox_tuple] = f_name

        for item in raw_items:
            pts = np.array(item["bbox"], dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)

            is_highlighted = tuple(item["bbox"][0]) in highlight_boxes
            box_color = (0, 200, 80) if is_highlighted else (255, 140, 0)
            thickness = 2 if is_highlighted else 1

            cv2.polylines(doc_overlay, [pts], isClosed=True, color=box_color, thickness=thickness)

        # Right Panel dimensions
        card_w = 640
        canvas_h = max(target_doc_h, 750)
        canvas_w = target_doc_w + card_w + 30

        canvas = np.full((canvas_h, canvas_w, 3), 26, dtype=np.uint8)

        # Place document in canvas (centered vertically on left)
        y_offset_doc = (canvas_h - target_doc_h) // 2
        canvas[y_offset_doc:y_offset_doc + target_doc_h, 15:15 + target_doc_w] = doc_overlay

        # Draw decorative border around document
        cv2.rectangle(
            canvas,
            (14, y_offset_doc - 1),
            (15 + target_doc_w, y_offset_doc + target_doc_h),
            (70, 70, 70),
            1
        )

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        def get_font(size: int, bold: bool = False):
            font_paths = [
                "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
            ]
            for p in font_paths:
                if os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        title_font = get_font(22, bold=True)
        subtitle_font = get_font(13, bold=False)
        section_font = get_font(14, bold=True)
        key_font = get_font(13, bold=True)
        val_font = get_font(14, bold=False)
        meta_font = get_font(11, bold=False)

        card_x1 = target_doc_w + 30
        card_y1 = 20
        card_x2 = canvas_w - 20
        card_y2 = canvas_h - 20

        # Draw rounded card panel
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=12,
            fill=(34, 39, 46),
            outline=(68, 76, 86),
            width=1
        )

        # Header Badge & Title
        draw.rounded_rectangle([card_x1 + 20, card_y1 + 18, card_x1 + 130, card_y1 + 40], radius=5, fill=(31, 111, 235))
        draw.text((card_x1 + 30, card_y1 + 22), "OCR SCREENING", font=meta_font, fill=(255, 255, 255))

        draw.text((card_x1 + 20, card_y1 + 50), "Document Information", font=title_font, fill=(240, 246, 252))

        doc_title = parsed_fields["document_title"]["value"] or "Standard Identity Document"
        draw.text((card_x1 + 20, card_y1 + 82), f"Source File: {image_path.name}  |  {doc_title}", font=subtitle_font, fill=(139, 148, 158))

        # Divider line
        draw.line([(card_x1 + 20, card_y1 + 105), (card_x2 - 20, card_y1 + 105)], fill=(48, 54, 61), width=1)

        fields_to_render = [
            ("Full Name", parsed_fields["full_name"]),
            ("Surname", parsed_fields["surname"]),
            ("Given Names", parsed_fields["given_names"]),
            ("Personal / National ID", parsed_fields["personal_number"]),
            ("Document / Card Number", parsed_fields["document_number"]),
            ("Date of Birth", parsed_fields["date_of_birth"]),
            ("Date of Issue", parsed_fields["date_of_issue"]),
            ("Date of Expiry", parsed_fields["date_of_expiry"]),
            ("Sex / Gender", parsed_fields["sex"]),
            ("Nationality", parsed_fields["nationality"]),
            ("Place of Birth", parsed_fields["place_of_birth"]),
            ("Issuing Authority", parsed_fields["authority"]),
        ]

        if parsed_fields["email"]["value"]:
            fields_to_render.append(("Email Address", parsed_fields["email"]))

        curr_y = card_y1 + 120
        row_height = 36

        for label, f_info in fields_to_render:
            val = f_info.get("value")
            conf = f_info.get("confidence", 0.0)

            draw.text((card_x1 + 22, curr_y), label, font=key_font, fill=(139, 148, 158))

            if val:
                val_text = str(val)
                draw.text((card_x1 + 210, curr_y), val_text, font=val_font, fill=(240, 246, 252))

                conf_pct = int(conf * 100) if conf <= 1.0 else int(conf)
                badge_bg = (35, 134, 54) if conf_pct >= 85 else ((187, 128, 9) if conf_pct >= 60 else (218, 54, 51))
                badge_w = 48
                draw.rounded_rectangle(
                    [card_x2 - 25 - badge_w, curr_y - 1, card_x2 - 25, curr_y + 18],
                    radius=4,
                    fill=badge_bg
                )
                draw.text((card_x2 - 20 - badge_w, curr_y + 1), f"{conf_pct}%", font=meta_font, fill=(255, 255, 255))
            else:
                draw.text((card_x1 + 210, curr_y), "Not detected / N/A", font=val_font, fill=(90, 95, 105))

            draw.line([(card_x1 + 20, curr_y + 26), (card_x2 - 20, curr_y + 26)], fill=(40, 45, 52), width=1)
            curr_y += row_height

        mrz_data = parsed_fields.get("mrz_lines", {}).get("value", [])
        if mrz_data:
            curr_y += 10
            draw.text((card_x1 + 22, curr_y), "Machine Readable Zone (MRZ)", font=section_font, fill=(88, 166, 255))
            curr_y += 24
            for mrz_line in mrz_data:
                draw.text((card_x1 + 22, curr_y), mrz_line, font=meta_font, fill=(126, 231, 135))
                curr_y += 18

        total_detected = len(raw_items)
        parsed_count = sum(1 for _, info in fields_to_render if info.get("value"))
        draw.text(
            (card_x1 + 22, card_y2 - 28),
            f"Extraction Stats: {parsed_count}/{len(fields_to_render)} key fields identified ({total_detected} total OCR blocks)",
            font=meta_font,
            fill=(110, 118, 129)
        )

        pil_img.save(str(output_path), quality=95)
        return output_path

    def process_file(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        save_visual: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image file through OCR, field extraction, and visualization.
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = Path("ocr_output")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_items = self.extract_raw_ocr(image_path)
        parsed_fields = self.parse_document_fields(raw_items)

        visual_path = None
        if save_visual:
            out_img_name = f"annotated_{image_path.stem}.png"
            visual_path = output_dir / out_img_name
            self.generate_visual_dashboard(image_path, raw_items, parsed_fields, visual_path)

        result_payload = {
            "file": str(image_path.resolve()),
            "total_blocks": len(raw_items),
            "parsed_fields": {
                k: v["value"] for k, v in parsed_fields.items() if v["value"] is not None
            },
            "field_confidences": {
                k: round(v["confidence"], 3) for k, v in parsed_fields.items() if v["value"] is not None
            },
            "raw_ocr": [
                {"text": it["text"], "confidence": round(it["confidence"], 3), "bbox": it["bbox"]}
                for it in raw_items
            ],
            "visual_dashboard": str(visual_path.resolve()) if visual_path else None
        }

        return result_payload


def print_formatted_summary(result: Dict[str, Any]):
    """
    Print a neat terminal summary table.
    """
    file_name = Path(result["file"]).name
    fields = result["parsed_fields"]
    confs = result["field_confidences"]

    print("\n" + "=" * 70)
    print(f" DOCUMENT OCR EXTRACTION SUMMARY: {file_name}")
    print("=" * 70)
    print(f"{'FIELD':<26} | {'VALUE':<30} | {'CONF'}")
    print("-" * 70)

    display_keys = [
        ("Document Title", "document_title"),
        ("Full Name", "full_name"),
        ("Surname", "surname"),
        ("Given Names", "given_names"),
        ("Personal ID / SSN", "personal_number"),
        ("Card / Document No", "document_number"),
        ("Date of Birth", "date_of_birth"),
        ("Date of Issue", "date_of_issue"),
        ("Date of Expiry", "date_of_expiry"),
        ("Sex / Gender", "sex"),
        ("Nationality", "nationality"),
        ("Place of Birth", "place_of_birth"),
        ("Issuing Authority", "authority"),
        ("Email Address", "email"),
    ]

    for label, k in display_keys:
        if k in fields:
            val = str(fields[k])
            conf_str = f"{int(confs.get(k, 1.0) * 100)}%"
            print(f"{label:<26} | {val:<30} | {conf_str}")

    if "mrz_lines" in fields:
        print("-" * 70)
        print("MRZ Lines:")
        for line in fields["mrz_lines"]:
            print(f"  > {line}")

    if result.get("visual_dashboard"):
        print("-" * 70)
        print(f"Annotated Visual Dashboard: {result['visual_dashboard']}")
    print("=" * 70 + "\n")


def find_sample_images(root: Path) -> List[Tuple[str, Path]]:
    """
    Locate representative samples across SIDTD, MIDV_2020, FCD-V, and uploads.
    """
    samples = []

    sidtd_alb = root / "data" / "SIDTD" / "templates" / "Images" / "reals" / "alb_id_00.jpg"
    if sidtd_alb.exists():
        samples.append(("SIDTD Albanian ID (Real)", sidtd_alb))

    sidtd_fake = root / "data" / "SIDTD" / "templates" / "Images" / "fakes" / "alb_id_00_fake_1_0.jpg"
    if sidtd_fake.exists():
        samples.append(("SIDTD Albanian ID (Fake)", sidtd_fake))

    fcdv_can = root / "data" / "FCD-V" / "canada" / "doc.1.jpg"
    if fcdv_can.exists():
        samples.append(("FCD-V Canada ID", fcdv_can))

    midv_alb = root / "data" / "MIDV_2020" / "photo" / "images" / "alb_id" / "00.jpg"
    if midv_alb.exists():
        samples.append(("MIDV-2020 Photo (alb_id/00.jpg)", midv_alb))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Document OCR & Information Extraction")
    parser.add_argument("--image", type=str, help="Path to input document image")
    parser.add_argument("--dataset", type=str, choices=["sidtd", "midv2020", "fcd-v", "all"], help="Process dataset images")
    parser.add_argument("--max-samples", type=int, default=3, help="Max samples to process per dataset")
    parser.add_argument("--output-dir", type=str, default="ocr_output", help="Directory to save visual output cards")
    parser.add_argument("--test-sample", action="store_true", help="Run automated test on sample documents")
    parser.add_argument("--json", action="store_true", help="Print json formatted result")

    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent

    ocr_engine = DocumentOCR()

    if args.test_sample or (not args.image and not args.dataset):
        print("Running OCR on representative sample documents...")
        samples = find_sample_images(root_dir)
        for label, img_path in samples:
            print(f"\nProcessing [{label}] -> {img_path.name}")
            res = ocr_engine.process_file(img_path, output_dir=Path(args.output_dir))
            print_formatted_summary(res)
        return

    if args.image:
        img_p = Path(args.image)
        if not img_p.is_absolute():
            img_p = root_dir / img_p
        res = ocr_engine.process_file(img_p, output_dir=Path(args.output_dir))
        if args.json:
            import json
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print_formatted_summary(res)
        return

    if args.dataset:
        datasets_to_run = []
        if args.dataset in ["sidtd", "all"]:
            datasets_to_run.append(("SIDTD", root_dir / "data" / "SIDTD" / "templates" / "Images" / "reals"))
        if args.dataset in ["fcd-v", "all"]:
            datasets_to_run.append(("FCD-V", root_dir / "data" / "FCD-V" / "canada"))
        if args.dataset in ["midv2020", "all"]:
            datasets_to_run.append(("MIDV_2020", root_dir / "data" / "MIDV_2020" / "photo" / "images" / "alb_id"))

        for d_name, d_path in datasets_to_run:
            if not d_path.exists():
                print(f"Directory {d_path} not found.")
                continue
            images = list(d_path.glob("*.jpg")) + list(d_path.glob("*.png"))
            images = images[:args.max_samples]
            print(f"\nProcessing dataset {d_name} ({len(images)} images)...")
            for img_p in images:
                res = ocr_engine.process_file(img_p, output_dir=Path(args.output_dir) / d_name)
                print_formatted_summary(res)


if __name__ == "__main__":
    main()
