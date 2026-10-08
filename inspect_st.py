import os, re
import streamlit

st_dir = os.path.dirname(streamlit.__file__)
js_file = os.path.join(st_dir, 'static', 'static', 'js', 'index.CKCTizkM.js')
with open(js_file, 'r', encoding='utf-8', errors='ignore') as f:
    code = f.read()

# find where R is defined. In React JSX, c(R, ...) means R is a function/component.
# Let's search for "R=" or "function R("
# Or search for "material-symbols" or "material" in the whole bundle
for m in re.finditer(r':material/(\w+):', code):
    print("Found iconValue shortcode:", m.group(0))

# Search for where :material/ is parsed / stripped / converted to icon text
for m in re.finditer(r'replace\([^\)]*material[^\)]*\)', code):
    print("Replace material match:", m.group(0))

pos = code.find('material-symbols')
print("material-symbols position:", pos)
if pos != -1:
    print(code[pos-100:pos+200])
