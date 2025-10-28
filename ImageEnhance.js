//CA
import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download, RefreshCw, Sliders } from 'lucide-react';

export default function AIImageEnhancer() {
  const [originalImage, setOriginalImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const [settings, setSettings] = useState({
    brightness: 0,
    contrast: 0,
    saturation: 0,
    sharpness: 0,
    highlights: 0,
    shadows: 0,
    warmth: 0,
    vignette: 0
  });

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setOriginalImage(img);
          setEnhancedImage(null);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const autoEnhance = () => {
    if (!originalImage) return;
    setIsProcessing(true);

    setTimeout(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      canvas.width = originalImage.width;
      canvas.height = originalImage.height;
      ctx.drawImage(originalImage, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Analyze image for auto-adjustments
      const analysis = analyzeImage(data);
      
      // Apply auto-enhancement based on analysis
      const enhanced = applyEnhancements(data, {
        brightness: analysis.brightness,
        contrast: analysis.contrast,
        saturation: analysis.saturation,
        sharpness: 0.3,
        highlights: analysis.highlights,
        shadows: analysis.shadows,
        warmth: 0,
        vignette: 0.1
      }, canvas.width, canvas.height);

      ctx.putImageData(enhanced, 0, 0);
      setEnhancedImage(canvas.toDataURL());
      setIsProcessing(false);
    }, 100);
  };

  const analyzeImage = (data) => {
    let totalBrightness = 0;
    let totalSaturation = 0;
    const histogram = new Array(256).fill(0);

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      const brightness = (r + g + b) / 3;
      totalBrightness += brightness;
      histogram[Math.floor(brightness)]++;
      
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const saturation = max === 0 ? 0 : (max - min) / max;
      totalSaturation += saturation;
    }

    const pixelCount = data.length / 4;
    const avgBrightness = totalBrightness / pixelCount;
    const avgSaturation = totalSaturation / pixelCount;

    // Calculate adjustments
    const brightnessAdjust = (128 - avgBrightness) / 255 * 0.3;
    const contrastAdjust = avgBrightness < 100 || avgBrightness > 155 ? 0.2 : 0.1;
    const saturationAdjust = avgSaturation < 0.3 ? 0.2 : 0;

    // Calculate highlights and shadows
    let darkPixels = 0, brightPixels = 0;
    for (let i = 0; i < 64; i++) darkPixels += histogram[i];
    for (let i = 192; i < 256; i++) brightPixels += histogram[i];
    
    const shadowsAdjust = (darkPixels / pixelCount) > 0.2 ? 0.3 : 0;
    const highlightsAdjust = (brightPixels / pixelCount) > 0.2 ? -0.2 : 0;

    return {
      brightness: brightnessAdjust,
      contrast: contrastAdjust,
      saturation: saturationAdjust,
      highlights: highlightsAdjust,
      shadows: shadowsAdjust
    };
  };

  const applyEnhancements = (data, settings, width, height) => {
    const enhanced = new ImageData(new Uint8ClampedArray(data), width, height);
    const pixels = enhanced.data;

    // First pass: brightness, contrast, saturation, highlights, shadows
    for (let i = 0; i < pixels.length; i += 4) {
      let r = pixels[i];
      let g = pixels[i + 1];
      let b = pixels[i + 2];

      // Brightness
      if (settings.brightness !== 0) {
        const adjust = settings.brightness * 50;
        r += adjust;
        g += adjust;
        b += adjust;
      }

      // Contrast
      if (settings.contrast !== 0) {
        const factor = (1 + settings.contrast);
        r = ((r - 128) * factor) + 128;
        g = ((g - 128) * factor) + 128;
        b = ((b - 128) * factor) + 128;
      }

      // Highlights and Shadows
      const brightness = (r + g + b) / 3;
      if (settings.highlights !== 0 && brightness > 170) {
        const factor = 1 - (settings.highlights * 0.5);
        r *= factor;
        g *= factor;
        b *= factor;
      }
      if (settings.shadows !== 0 && brightness < 85) {
        const factor = 1 + (settings.shadows * 0.5);
        r *= factor;
        g *= factor;
        b *= factor;
      }

      // Saturation
      if (settings.saturation !== 0) {
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        const factor = 1 + settings.saturation;
        r = gray + (r - gray) * factor;
        g = gray + (g - gray) * factor;
        b = gray + (b - gray) * factor;
      }

      // Warmth (color temperature)
      if (settings.warmth !== 0) {
        r += settings.warmth * 20;
        b -= settings.warmth * 20;
      }

      pixels[i] = Math.max(0, Math.min(255, r));
      pixels[i + 1] = Math.max(0, Math.min(255, g));
      pixels[i + 2] = Math.max(0, Math.min(255, b));
    }

    // Sharpening pass
    if (settings.sharpness > 0) {
      applySharpen(pixels, width, height, settings.sharpness);
    }

    // Vignette
    if (settings.vignette > 0) {
      applyVignette(pixels, width, height, settings.vignette);
    }

    return enhanced;
  };

  const applySharpen = (pixels, width, height, amount) => {
    const tempPixels = new Uint8ClampedArray(pixels);
    const strength = amount * 2;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        for (let c = 0; c < 3; c++) {
          const center = tempPixels[idx + c];
          const top = tempPixels[((y - 1) * width + x) * 4 + c];
          const bottom = tempPixels[((y + 1) * width + x) * 4 + c];
          const left = tempPixels[(y * width + (x - 1)) * 4 + c];
          const right = tempPixels[(y * width + (x + 1)) * 4 + c];
          
          const sharpened = center * (1 + 4 * strength) - (top + bottom + left + right) * strength;
          pixels[idx + c] = Math.max(0, Math.min(255, sharpened));
        }
      }
    }
  };

  const applyVignette = (pixels, width, height, amount) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const maxDist = Math.sqrt(centerX * centerX + centerY * centerY);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const dx = x - centerX;
        const dy = y - centerY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const factor = 1 - (dist / maxDist) * amount;

        pixels[idx] *= factor;
        pixels[idx + 1] *= factor;
        pixels[idx + 2] *= factor;
      }
    }
  };

  const applyManualAdjustments = () => {
    if (!originalImage) return;
    setIsProcessing(true);

    setTimeout(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      canvas.width = originalImage.width;
      canvas.height = originalImage.height;
      ctx.drawImage(originalImage, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const enhanced = applyEnhancements(imageData.data, settings, canvas.width, canvas.height);

      ctx.putImageData(enhanced, 0, 0);
      setEnhancedImage(canvas.toDataURL());
      setIsProcessing(false);
    }, 100);
  };

  const downloadImage = () => {
    if (!enhancedImage) return;
    const link = document.createElement('a');
    link.download = 'enhanced-image.png';
    link.href = enhancedImage;
    link.click();
  };

  const resetSettings = () => {
    setSettings({
      brightness: 0,
      contrast: 0,
      saturation: 0,
      sharpness: 0,
      highlights: 0,
      shadows: 0,
      warmth: 0,
      vignette: 0
    });
  };

  useEffect(() => {
    if (showAdvanced && originalImage) {
      applyManualAdjustments();
    }
  }, [settings]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">AI Image Enhancer</h1>
          <p className="text-purple-200">Professional-grade image enhancement powered by intelligent algorithms</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Original Image */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">Original</h2>
            <div className="bg-slate-800 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
              {originalImage ? (
                <img src={originalImage.src} alt="Original" className="max-w-full max-h-full object-contain" />
              ) : (
                <div className="text-center text-gray-400">
                  <Upload className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Upload an image to get started</p>
                </div>
              )}
            </div>
          </div>

          {/* Enhanced Image */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">Enhanced</h2>
            <div className="bg-slate-800 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
              {enhancedImage ? (
                <img src={enhancedImage} alt="Enhanced" className="max-w-full max-h-full object-contain" />
              ) : (
                <div className="text-center text-gray-400">
                  <Sliders className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Enhanced image will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
          <div className="flex flex-wrap gap-4 mb-6">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors"
            >
              <Upload className="w-5 h-5" />
              Upload Image
            </button>
            
            <button
              onClick={autoEnhance}
              disabled={!originalImage || isProcessing}
              className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
            >
              <RefreshCw className={`w-5 h-5 ${isProcessing ? 'animate-spin' : ''}`} />
              Auto Enhance
            </button>

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              <Sliders className="w-5 h-5" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced
            </button>

            <button
              onClick={downloadImage}
              disabled={!enhancedImage}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors ml-auto"
            >
              <Download className="w-5 h-5" />
              Download
            </button>
          </div>

          {/* Advanced Controls */}
          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t border-white/20">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-white">Manual Adjustments</h3>
                <button
                  onClick={resetSettings}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm transition-colors"
                >
                  Reset All
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(settings).map(([key, value]) => (
                  <div key={key}>
                    <div className="flex justify-between text-white mb-2">
                      <label className="capitalize">{key}</label>
                      <span className="text-purple-300">{value.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="-1"
                      max="1"
                      step="0.01"
                      value={value}
                      onChange={(e) => setSettings(prev => ({...prev, [key]: parseFloat(e.target.value)}))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}