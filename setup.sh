#!/bin/bash
# https://github.com/facebookresearch/how-to-autorl
set -e

pip install -e .[dehb]

pip uninstall dehb -y
pip install dehb==0.0.5
pip uninstall scipy -y
pip install scipy==1.12.0

echo "Installation complete."


# Replace tree.map > tree_util.tree_map
files=(
    "/home/4thhia/.local/lib/python3.10/site-packages/brax/io/mjcf.py"
    "/home/4thhia/.local/lib/python3.10/site-packages/brax/scan.py"
    "/home/4thhia/.local/lib/python3.10/site-packages/brax/generalized/dynamics.py"
    "/home/4thhia/.local/lib/python3.10/site-packages/brax/generalized/constraint.py"
)

for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    sed -i 's/tree\.map/tree_util.tree_map/g' "$file"
    echo "Processed: $file"
  else
    echo "File not found: $file"
  fi
done