PROJECT_DIR="/mnt/c/Users/nithi/Desktop/Nithin/DL Projects/FashionNet"
echo "=== Fixing numpy binary incompatibility ==="
pip install --upgrade numpy pandas opencv-python-headless pillow tqdm
echo ""
echo "=== Running dataset processing ==="
python "$PROJECT_DIR/ImageProcessing/dataset.py" \
  -i <Dataset path> \
       -l <CSV file path> \
       -o <Path to save processed images>
