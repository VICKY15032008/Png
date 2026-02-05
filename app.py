#!/usr/bin/env python3
"""
Flask Web Application for Unified PNG Compressor
Features: Upload, compress, side-by-side comparison, histograms, metrics
"""

import io, json, struct, hashlib, sys, os, shutil, tempfile, base64, pickle, lzma
import numpy as np
import zlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter, defaultdict
from skimage.metrics import structural_similarity as ssim
from flask import Flask, render_template, request, jsonify, send_file, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================================= FLASK APP SETUP =================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "unified-png-compressor-secret-key-2024")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# ================================= DATA STRUCTURES =================================
@dataclass
class CompressionResult:
    strategy_name: str
    output_path: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    psnr: float
    ssim_score: float
    hint_type: str
    params: dict
    
    @property
    def size_reduction_bytes(self):
        return self.original_size - self.compressed_size
    
    @property
    def is_improvement(self):
        return self.compressed_size < self.original_size

@dataclass
class UnifiedAnalysis:
    image_path: str
    width: int
    height: int
    original_size: int
    pixel_hash: str
    content_type: str
    hint_params: dict
    tile_count: int
    avg_variance: float
    avg_colors: float
    edge_density: float
    alpha_sparsity: float

# ================================= COMPRESSION FUNCTIONS =================================
def analyze_content_fast(image_path: str) -> Tuple[str, dict, UnifiedAnalysis]:
    """Fast unified content analysis"""
    img = Image.open(image_path).convert("RGBA")
    pixels = np.asarray(img)
    h, w = pixels.shape[:2]
    
    tile_size = min(64, h//4, w//4) if h > 64 and w > 64 else 32
    features = []
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            th, tw = min(tile_size, h-y), min(tile_size, w-x)
            tile = pixels[y:y+th, x:x+tw]
            gray = np.mean(tile[:,:,0:3], axis=2)
            
            features.append({
                'variance': float(np.var(gray)),
                'edge_density': float(np.mean(np.abs(np.diff(gray)))),
                'colors': len(np.unique(gray.astype(np.uint8))),
                'alpha_sparse': float(np.sum(tile[:,:,3] == 0) / (th*tw))
            })
    
    avg_var = np.mean([f['variance'] for f in features])
    avg_colors = np.mean([f['colors'] for f in features])
    avg_edge = np.mean([f['edge_density'] for f in features])
    avg_alpha = np.mean([f['alpha_sparse'] for f in features])
    
    if avg_var < 300 and avg_colors < 24:
        content_type = 'icon'
        hint_params = {'colors': 32, 'aggressive': True, 'palette_colors': 8}
    elif avg_var < 1000:
        content_type = 'ui'
        hint_params = {'colors': 96, 'aggressive': True, 'palette_colors': 48}
    elif avg_colors < 32:
        content_type = 'chart'
        hint_params = {'colors': 64, 'aggressive': True, 'palette_colors': 16}
    else:
        content_type = 'photo'
        hint_params = {'colors': 192, 'aggressive': False, 'palette_colors': 96}
    
    pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    
    analysis = UnifiedAnalysis(
        image_path=image_path,
        width=w,
        height=h,
        original_size=os.path.getsize(image_path),
        pixel_hash=pixel_hash[:16],
        content_type=content_type,
        hint_params=hint_params,
        tile_count=len(features),
        avg_variance=avg_var,
        avg_colors=avg_colors,
        edge_density=avg_edge,
        alpha_sparsity=avg_alpha
    )
    
    return content_type, hint_params, analysis

def compute_quality_metrics(orig_path: str, comp_path: str) -> Tuple[float, float]:
    """Compute PSNR and SSIM between original and compressed"""
    orig_img = Image.open(orig_path).convert("RGB")
    comp_img = Image.open(comp_path).convert("RGB")
    
    orig_array = np.asarray(orig_img)
    comp_array = np.asarray(comp_img)
    
    mse = np.mean((orig_array.astype(np.float32) - comp_array.astype(np.float32))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    try:
        h, w = orig_array.shape[:2]
        win_size = min(7, min(h, w) - 2) // 2 * 2 + 1
        if win_size < 3:
            win_size = 3
        ssim_val = ssim(orig_array, comp_array, channel_axis=-1, data_range=255, win_size=win_size)
    except:
        ssim_val = 1.0 / (1.0 + mse / (255**2))
    
    return psnr, ssim_val

def strategy1_smart_adaptive(input_path: str, content_type: str, hint_params: dict, temp_dir: str) -> CompressionResult:
    """Smart adaptive compression"""
    orig_size = os.path.getsize(input_path)
    orig_img = Image.open(input_path)
    
    best_size = orig_size
    best_path = input_path
    best_method = "original"
    
    # Method 1: Palette-based
    temp1 = os.path.join(temp_dir, 'strategy1_method1.png')
    try:
        img = orig_img.convert("RGBA")
        if hint_params['aggressive']:
            rgb = img.convert("RGB")
            paletted = rgb.convert("P", palette=Image.ADAPTIVE, colors=hint_params['colors'])
            optimized = paletted.convert("RGBA")
        else:
            optimized = img.convert("RGB")
        optimized.save(temp1, optimize=True, compress_level=9)
        size1 = os.path.getsize(temp1)
        if size1 < best_size:
            best_size, best_path, best_method = size1, temp1, "palette_adaptive"
    except:
        pass
    
    # Method 2: Bit-depth reduction
    temp2 = os.path.join(temp_dir, 'strategy1_method2.png')
    try:
        img = orig_img.convert("RGBA")
        reduced = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        reduced.save(temp2, optimize=True, compress_level=9)
        size2 = os.path.getsize(temp2)
        if size2 < best_size:
            best_size, best_path, best_method = size2, temp2, "bit_depth_reduction"
    except:
        pass
    
    # Method 3: RGBA optimization
    temp3 = os.path.join(temp_dir, 'strategy1_method3.png')
    try:
        img = orig_img.convert("RGBA")
        img.save(temp3, optimize=True, compress_level=9)
        size3 = os.path.getsize(temp3)
        if size3 < best_size:
            best_size, best_path, best_method = size3, temp3, "rgba_optimized"
    except:
        pass
    
    output_path = os.path.join(temp_dir, 'strategy1_output.png')
    if best_path != input_path:
        shutil.copy2(best_path, output_path)
        final_size = best_size
        psnr, ssim_val = compute_quality_metrics(input_path, output_path)
    else:
        shutil.copy2(input_path, output_path)
        final_size = orig_size
        psnr, ssim_val = float('inf'), 1.0
    
    cr = max(0, (1 - final_size / orig_size) * 100)
    
    return CompressionResult(
        strategy_name="Smart Adaptive",
        output_path=output_path,
        original_size=orig_size,
        compressed_size=final_size,
        compression_ratio=cr,
        psnr=psnr,
        ssim_score=ssim_val,
        hint_type=content_type,
        params={'method': best_method, **hint_params}
    )

def strategy2_visual_lossless(input_path: str, content_type: str, hint_params: dict, temp_dir: str) -> CompressionResult:
    """Visual lossless compression"""
    orig_size = os.path.getsize(input_path)
    orig_img = Image.open(input_path).convert("RGB")
    palette_colors = hint_params.get('palette_colors', 64)
    
    output_path = os.path.join(temp_dir, 'strategy2_output.png')
    quant_img = orig_img.convert("P", palette=Image.ADAPTIVE, colors=palette_colors)
    quant_img.save(output_path, optimize=True, compress_level=9)
    
    final_size = os.path.getsize(output_path)
    psnr, ssim_val = compute_quality_metrics(input_path, output_path)
    cr = max(0, (1 - final_size / orig_size) * 100)
    
    return CompressionResult(
        strategy_name="Visual Lossless",
        output_path=output_path,
        original_size=orig_size,
        compressed_size=final_size,
        compression_ratio=cr,
        psnr=psnr,
        ssim_score=ssim_val,
        hint_type=content_type,
        params={'palette_colors': palette_colors, 'method': 'adaptive_palette'}
    )

def strategy3_hybrid(input_path: str, content_type: str, hint_params: dict, temp_dir: str) -> CompressionResult:
    """Hybrid multi-strategy"""
    orig_size = os.path.getsize(input_path)
    orig_img = Image.open(input_path)
    
    best_size = orig_size
    best_path = input_path
    best_params = {}
    
    color_depths = [8, 16, 32, 48, 64, 96, 128, 192, 256]
    
    for colors in color_depths:
        temp_path = os.path.join(temp_dir, f'strategy3_colors{colors}.png')
        try:
            img_rgb = orig_img.convert("RGB")
            quant = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=colors)
            quant.save(temp_path, optimize=True, compress_level=9)
            size = os.path.getsize(temp_path)
            if size < best_size:
                best_size = size
                best_path = temp_path
                best_params = {'colors': colors, 'mode': 'RGB_palette'}
        except:
            pass
    
    temp_rgba = os.path.join(temp_dir, 'strategy3_rgba.png')
    try:
        img_rgba = orig_img.convert("RGBA")
        img_rgba.save(temp_rgba, optimize=True, compress_level=9)
        size = os.path.getsize(temp_rgba)
        if size < best_size:
            best_size = size
            best_path = temp_rgba
            best_params = {'mode': 'RGBA_optimized'}
    except:
        pass
    
    output_path = os.path.join(temp_dir, 'strategy3_output.png')
    if best_path != input_path:
        shutil.copy2(best_path, output_path)
        final_size = best_size
        psnr, ssim_val = compute_quality_metrics(input_path, output_path)
    else:
        shutil.copy2(input_path, output_path)
        final_size = orig_size
        psnr, ssim_val = float('inf'), 1.0
    
    cr = max(0, (1 - final_size / orig_size) * 100)
    
    return CompressionResult(
        strategy_name="Hybrid Multi-Strategy",
        output_path=output_path,
        original_size=orig_size,
        compressed_size=final_size,
        compression_ratio=cr,
        psnr=psnr,
        ssim_score=ssim_val,
        hint_type=content_type,
        params=best_params
    )

# ================================= VISUALIZATION HELPERS =================================
def generate_histogram(image_path: str, title: str) -> str:
    """Generate histogram and return as base64"""
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    colors = ['red', 'green', 'blue']
    channels = ['Red', 'Green', 'Blue']
    
    for i, (ax, color, channel) in enumerate(zip(axes, colors, channels)):
        ax.hist(arr[:,:,i].flatten(), bins=256, color=color, alpha=0.7, range=(0, 256))
        ax.set_title(f'{channel} Channel')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()

def image_to_base64(image_path: str) -> str:
    """Convert image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# ================================= FLASK ROUTES =================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Only PNG and JPEG files are supported'}), 400
    
    try:
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.png')
        file.save(upload_path)
        
        # Convert to PNG if needed
        img = Image.open(upload_path)
        if img.format != 'PNG':
            img.save(upload_path, 'PNG')
        
        # Analyze image
        content_type, hint_params, analysis = analyze_content_fast(upload_path)
        
        # Create temp directory for compression
        temp_dir = tempfile.mkdtemp()
        results = []
        
        # Run all strategies
        try:
            result1 = strategy1_smart_adaptive(upload_path, content_type, hint_params, temp_dir)
            results.append(result1)
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
        
        try:
            result2 = strategy2_visual_lossless(upload_path, content_type, hint_params, temp_dir)
            results.append(result2)
        except Exception as e:
            print(f"Strategy 2 failed: {e}")
        
        try:
            result3 = strategy3_hybrid(upload_path, content_type, hint_params, temp_dir)
            results.append(result3)
        except Exception as e:
            print(f"Strategy 3 failed: {e}")
        
        if not results:
            return jsonify({'error': 'All compression strategies failed'}), 500
        
        # Find best result
        valid_results = [r for r in results if r.is_improvement]
        if not valid_results:
            best_result = min(results, key=lambda r: r.compressed_size)
        else:
            best_result = max(valid_results, key=lambda r: (r.compression_ratio, r.ssim_score))
        
        # Generate histograms
        hist_original = generate_histogram(upload_path, 'Original Image Histogram')
        hist_compressed = generate_histogram(best_result.output_path, 'Compressed Image Histogram')
        
        # Convert images to base64
        img_original = image_to_base64(upload_path)
        img_compressed = image_to_base64(best_result.output_path)
        
        # Save best compressed image for download
        compressed_download_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_best.png')
        shutil.copy2(best_result.output_path, compressed_download_path)
        
        # Prepare response
        response_data = {
            'success': True,
            'analysis': {
                'content_type': analysis.content_type,
                'dimensions': f"{analysis.width}x{analysis.height}",
                'original_size': analysis.original_size,
                'avg_variance': round(analysis.avg_variance, 2),
                'avg_colors': round(analysis.avg_colors, 2),
                'edge_density': round(analysis.edge_density, 4)
            },
            'images': {
                'original': img_original,
                'compressed': img_compressed
            },
            'histograms': {
                'original': hist_original,
                'compressed': hist_compressed
            },
            'best_result': {
                'strategy': best_result.strategy_name,
                'original_size': best_result.original_size,
                'compressed_size': best_result.compressed_size,
                'size_reduction': best_result.size_reduction_bytes,
                'compression_ratio': round(best_result.compression_ratio, 2),
                'psnr': round(best_result.psnr, 2) if best_result.psnr != float('inf') else 'Perfect',
                'ssim': round(best_result.ssim_score, 4),
                'params': best_result.params
            },
            'all_strategies': [
                {
                    'name': r.strategy_name,
                    'size': r.compressed_size,
                    'compression_ratio': round(r.compression_ratio, 2),
                    'psnr': round(r.psnr, 2) if r.psnr != float('inf') else 'Perfect',
                    'ssim': round(r.ssim_score, 4),
                    'is_best': r == best_result
                }
                for r in results
            ]
        }
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Compression failed: {str(e)}'}), 500

@app.route('/download')
def download():
    compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_best.png')
    if os.path.exists(compressed_path):
        return send_file(compressed_path, as_attachment=True, download_name='compressed_image.png')
    return jsonify({'error': 'No compressed image available'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
