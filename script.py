import zipfile
import os
from lxml import etree
import shutil

INPUT_DOCX = "input.docx"
OUTPUT_DOCX = "output.docx"
TEMP_DIR = "temp_docx"

REPLACEMENTS = [
    ("Keshav Gupta", "Aaradhya Bhardwaj"),
    ("Keshav", "Aaradhya"),
    ("24/SE/090", "24/SE/005"),
]

# Step 1: unzip
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

with zipfile.ZipFile(INPUT_DOCX, 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR)

# Step 2: process all XML files
for root, _, files in os.walk(TEMP_DIR):
    for file in files:
        if file.endswith(".xml"):
            path = os.path.join(root, file)

            try:
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(path, parser)
                root_elem = tree.getroot()

                # iterate all text nodes
                for elem in root_elem.iter():
                    if elem.text:
                        text = elem.text
                        for old, new in REPLACEMENTS:
                            text = text.replace(old, new)
                        elem.text = text

                    if elem.tail:
                        text = elem.tail
                        for old, new in REPLACEMENTS:
                            text = text.replace(old, new)
                        elem.tail = text

                tree.write(path, xml_declaration=True, encoding="UTF-8")

            except Exception as e:
                print(f"Skipped {path}: {e}")

# Step 3: zip back
with zipfile.ZipFile(OUTPUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as docx:
    for root, _, files in os.walk(TEMP_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, TEMP_DIR)
            docx.write(full_path, rel_path)

# cleanup
shutil.rmtree(TEMP_DIR)

print("Done. Output:", OUTPUT_DOCX)
