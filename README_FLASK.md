# 🎨 Unified PNG Compressor - Web Application

A beautiful, professional web application for advanced PNG image compression using multiple compression strategies.

## ✨ Features

### Compression Strategies
1. **Smart Adaptive** - Multiple techniques (palette, bit-depth, RGBA)
2. **Visual Lossless** - Content-aware palette quantization
3. **Hybrid Multi-Strategy** - Tests 8-256 color depths for optimal compression

### Web Interface Features
- 📤 **Drag & Drop Upload** - Easy file uploading
- 🔄 **Side-by-Side Comparison** - Visual comparison of original vs compressed
- 📊 **Color Histograms** - RGB channel distribution analysis
- 📈 **Quality Metrics** - PSNR and SSIM measurements
- 🏆 **Strategy Comparison** - See all strategies' performance
- ⬇️ **One-Click Download** - Download best compressed image
- 📱 **Responsive Design** - Works on mobile and desktop

## 🚀 Installation

### Requirements
```bash
pip install flask pillow numpy scikit-image matplotlib
```

### Quick Start
```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

## 📖 Usage

1. **Upload Image**
   - Click the upload box or drag & drop a PNG/JPEG file
   - Max file size: 16MB

2. **Compress**
   - Click "Compress Image" button
   - Wait for analysis and compression (3-10 seconds)

3. **Review Results**
   - View side-by-side comparison
   - Check quality metrics (PSNR, SSIM)
   - Examine color histograms
   - Compare all strategies

4. **Download**
   - Click "Download Compressed PNG" to save the best result

## 📊 Understanding the Metrics

### Compression Ratio
- Percentage of size reduction from original
- Higher is better (more compression)
- Example: 86.9% means 86.9% smaller than original

### PSNR (Peak Signal-to-Noise Ratio)
- Measures pixel-level accuracy in dB
- **40+ dB**: Excellent - virtually identical
- **30-40 dB**: Very good - minor differences
- **20-30 dB**: Good - acceptable quality
- **<20 dB**: Fair - noticeable artifacts

### SSIM (Structural Similarity Index)
- Measures perceptual similarity (0.0 to 1.0)
- **0.95-1.00**: Excellent - perceptually identical
- **0.90-0.95**: Very good
- **0.80-0.90**: Good - acceptable for most uses
- **<0.80**: Fair to poor quality

## 🎯 Content Type Detection

The application automatically detects image content type:

- **Icon**: Low variance, few colors → Aggressive compression
- **UI**: Medium variance → Balanced compression
- **Chart**: Few distinct colors, high edges → Optimized compression
- **Photo**: High variance, many colors → Quality-focused compression

## 🏗️ Architecture

```
Flask Application
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Beautiful web interface
└── uploads/              # Temporary file storage
```

### Compression Pipeline

```
User Upload
    ↓
Image Analysis (Content Type Detection)
    ↓
Strategy 1: Smart Adaptive
├─ Palette compression
├─ Bit-depth reduction
└─ RGBA optimization
    ↓
Strategy 2: Visual Lossless
└─ Adaptive palette quantization
    ↓
Strategy 3: Hybrid Multi-Strategy
├─ Test 8-256 colors
└─ RGBA optimization
    ↓
Compare Results & Select Best
    ↓
Generate Visualizations
├─ Histograms (RGB channels)
├─ Quality metrics (PSNR/SSIM)
└─ Strategy comparison
    ↓
Display Results & Download
```

## 🎨 UI Components

### Header
- Gradient background (purple theme)
- Application title and description

### Upload Section
- Drag & drop zone with hover effects
- File type and size validation
- Visual feedback on file selection

### Results Section
1. **Image Analysis** - Content type, dimensions, statistics
2. **Side-by-Side Comparison** - Original vs compressed images
3. **Compression Metrics** - 4 metric cards with key stats
4. **Color Histograms** - RGB channel distributions
5. **Strategy Comparison** - Table showing all strategies
6. **Download Section** - Download and reset buttons

## 🔧 Customization

### Modify Compression Strategies

Edit `app.py` to adjust compression parameters:

```python
# In analyze_content_fast function
if avg_var < 300 and avg_colors < 24:
    content_type = 'icon'
    hint_params = {'colors': 32, 'aggressive': True, 'palette_colors': 8}
```

### Adjust Color Depths

Edit `strategy3_hybrid` function:

```python
color_depths = [8, 16, 32, 48, 64, 96, 128, 192, 256]
```

### Change UI Theme

Edit `templates/index.html` CSS:

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

## 📊 Example Results

### Test Case: UI Screenshot (512x512)

| Strategy | Size | Compression | PSNR | SSIM | Winner |
|----------|------|-------------|------|------|--------|
| Smart Adaptive | 61KB | 73.5% | 29.6dB | 0.98 | |
| Visual Lossless | 47KB | 79.6% | 23.8dB | 0.95 | |
| **Hybrid** | **30KB** | **86.9%** | 17.8dB | 0.82 | **🏆** |

**Original**: 232KB → **Best**: 30KB (87% smaller!)

## 🐛 Troubleshooting

### Error: "All compression strategies failed"
- Check if image is corrupted
- Verify image format (PNG/JPEG only)
- Try a different image

### Poor Compression Results
- Some images (photos, gradients) don't compress well
- Try different content types (icons, logos work best)
- Check quality metrics - high PSNR/SSIM means less compression

### Upload Fails
- Check file size (max 16MB)
- Verify file format
- Clear browser cache and retry

## 🔒 Security

- File size limit: 16MB
- Allowed formats: PNG, JPEG only
- Temporary files cleaned automatically
- No data persistence between sessions

## 🌐 Production Deployment

For production use:

1. **Set Secret Key**:
```bash
export SECRET_KEY='your-secret-key-here'
```

2. **Use Production Server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Enable HTTPS** with reverse proxy (nginx/Apache)

4. **Configure File Storage** for persistent uploads

## 📝 API Endpoints

### POST /compress
- **Input**: Form data with 'image' file
- **Output**: JSON with compression results
- **Response**: Images (base64), metrics, histograms

### GET /download
- **Output**: Best compressed PNG file
- **Filename**: compressed_image.png

## 🎯 Performance

- **Analysis Time**: 0.5-2 seconds
- **Compression Time**: 2-8 seconds (3 strategies)
- **Total Time**: 3-10 seconds per image
- **Memory Usage**: ~100-500MB per compression

## 📄 License

Free to use and modify for any purpose.

## 🙏 Credits

Based on unified PNG compression algorithms:
- Smart Adaptive Compression
- Visual Lossless Compression
- Hybrid Multi-Strategy Approach

Built with Flask, PIL, NumPy, scikit-image, and Matplotlib.

## 🔮 Future Enhancements

- [ ] Batch compression support
- [ ] WebP output format
- [ ] Custom compression presets
- [ ] Comparison with original tools
- [ ] Export compression report as PDF
- [ ] API key authentication
- [ ] Image history/gallery
- [ ] Advanced settings panel

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Unified PNG Compressor Team
