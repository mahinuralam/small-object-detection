# .gitignore Configuration for Small Object Detection Project

## 📋 What's Being Ignored

### 🗂️ **Heavy Files (NOT tracked in Git)**

#### **1. Datasets (2.2GB+)**
- `dataset/` - All VisDrone dataset files
- `data/` - All data directories
- Compressed files: `*.zip`, `*.tar.gz`, `*.rar`, etc.

#### **2. Model Checkpoints (300MB+ each)**
- `*.pth` - PyTorch models
- `*.pt` - PyTorch tensors
- `*.ckpt` - Training checkpoints
- `*.h5`, `*.keras` - TensorFlow/Keras models
- `*.onnx` - ONNX models

**Examples of ignored files:**
```
results/outputs/best_model.pth (316MB)
results/outputs/checkpoint_epoch_10.pth (316MB)
results/outputs_cddpa/checkpoint_epoch_12.pth (320MB)
models/reconstructor/best_reconstructor.pth (5MB)
```

#### **3. Training Results**
- `results/` - All training outputs
- `outputs*/` - Multiple output directories
- `logs/` - Training logs
- `*.log` - Log files

#### **4. Visualizations**
- `results/**/*.png` - Generated visualizations
- `results/**/*.jpg` - Detection images
- `*_visualization.png` - Result images

#### **5. Temporary & Cache Files**
- `__pycache__/` - Python bytecode
- `.ipynb_checkpoints/` - Jupyter checkpoints
- `tmp/`, `temp/` - Temporary directories

---

## ✅ **What IS Tracked in Git**

### **Tracked Files:**
- ✅ All Python source code (`*.py`)
- ✅ Configuration files (`*.yaml`, `*.json`)
- ✅ Documentation (`*.md`)
- ✅ Requirements files (`requirements*.txt`)
- ✅ Scripts and utilities
- ✅ Code structure (directory layout)
- ✅ Documentation images in `docs/`

---

## 🎯 **How to Track Specific Models (Optional)**

If you want to track a specific important model, you can:

### **Option 1: Uncommenting in .gitignore**
Edit `.gitignore` and uncomment these lines:
```gitignore
# !**/best_model.pth
# !models/reconstructor/best_reconstructor.pth
```

### **Option 2: Force add specific file**
```bash
git add -f results/outputs/best_model.pth
```

⚠️ **Warning**: Only do this for critical models < 100MB if absolutely necessary!

---

## 📦 **Recommended: Use Git LFS for Large Files**

For tracking large model files properly, use **Git Large File Storage (LFS)**:

### **Setup Git LFS:**
```bash
# Install Git LFS
sudo apt install git-lfs  # Ubuntu/Debian
# OR
brew install git-lfs      # macOS

# Initialize Git LFS in your repo
cd /home/mahin/Documents/notebook/small-object-detection
git lfs install

# Track specific file types
git lfs track "*.pth"
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

### **Add models with LFS:**
```bash
git add models/best_model.pth
git commit -m "Add best model via LFS"
```

---

## 📊 **File Size Reference**

| Category | Size | Status |
|----------|------|--------|
| **Dataset** | 2.2GB | ❌ Ignored |
| **Model checkpoints** | 316MB each | ❌ Ignored |
| **Training results** | ~1-2GB | ❌ Ignored |
| **Source code** | ~5MB | ✅ Tracked |
| **Documentation** | ~2MB | ✅ Tracked |

---

## 🔧 **Useful Git Commands**

### **Check what's being ignored:**
```bash
git status --ignored
```

### **Check size of tracked files:**
```bash
git ls-files | xargs ls -lh | awk '{sum+=$5} END {print sum}'
```

### **Clean untracked files (CAREFUL!):**
```bash
# Preview what will be deleted
git clean -dn

# Actually delete
git clean -df
```

### **Remove accidentally committed large files:**
```bash
# Remove from history (DANGEROUS - rewrites history!)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch results/outputs/best_model.pth' \
  --prune-empty --tag-name-filter cat -- --all
```

---

## 📝 **Best Practices**

1. **Never commit**:
   - Datasets (too large)
   - Model checkpoints (use LFS or external storage)
   - Generated visualizations (can be reproduced)

2. **Always commit**:
   - Source code
   - Configuration files
   - Training scripts
   - Documentation

3. **Use external storage** for:
   - Trained models → Google Drive, AWS S3, Hugging Face Hub
   - Datasets → Download links in README
   - Results → Zenodo, FigShare for papers

4. **Keep repository clean**:
   - Repository should be < 100MB
   - Model files should use Git LFS or external hosting
   - Include download scripts instead of files

---

## 📚 **External Storage Alternatives**

### **For Models:**
- **Hugging Face Hub**: https://huggingface.co/
- **Google Drive**: Shareable links
- **AWS S3**: Professional hosting
- **Dropbox**: Easy sharing
- **Zenodo**: Permanent DOI for publications

### **For Datasets:**
- **Google Drive**: Public sharing
- **Kaggle Datasets**: Public datasets
- **Zenodo**: Long-term preservation
- **Roboflow**: Computer vision datasets

### **Example README section:**
```markdown
## Download Trained Models

Download pre-trained models from:
- Baseline (38.02% mAP): [Google Drive Link]
- CD-DPA (SOTA): [Google Drive Link]
- Reconstructor: [Google Drive Link]

Extract to: `results/outputs/`
```

---

## 🚀 **Quick Start for New Users**

After cloning the repository:

```bash
# 1. Clone the repo (lightweight - no models/datasets)
git clone https://github.com/your-username/small-object-detection.git
cd small-object-detection

# 2. Download dataset (follow instructions in README)
# wget [dataset-link] -O dataset.zip
# unzip dataset.zip -d dataset/

# 3. Download models (follow instructions in README)  
# wget [model-link] -O best_model.pth
# mv best_model.pth results/outputs/

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start training/evaluation
python scripts/train/5_train_frcnn.py
```

---

## ❓ **FAQ**

**Q: Can I push my trained model to GitHub?**
A: Only if < 100MB and using Git LFS. Otherwise use external storage.

**Q: The .gitignore isn't working for files I already committed!**
A: Use `git rm --cached <file>` to untrack them first.

**Q: How do I share my results with collaborators?**
A: Use Google Drive/Dropbox for model files, commit only code and documentation.

**Q: Is the dataset included in the repo?**
A: No, it's ignored. Users must download separately (instructions in README).

---

**Last Updated**: February 26, 2026
**Project**: Small Object Detection for UAV Imagery (VisDrone Dataset)
