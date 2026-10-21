PROJECT_DIR="/mnt/c/Users/nithi/Desktop/Nithin/DL Projects/FashionNet"
echo "=== Fixing numpy binary incompatibility ==="
pip install --upgrade numpy pandas opencv-python-headless pillow tqdm
echo ""
echo "=== Running dataset processing ==="
python "$PROJECT_DIR/ImageProcessing/dataset.py" \
  -i "$PROJECT_DIR/train" \
  -l "$PROJECT_DIR/train.csv" \
  -o "$PROJECT_DIR/processed"
