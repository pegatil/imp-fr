// app.js - Aplicación completa para GitHub Pages

// ================== CONFIGURACIÓN ==================
const CLASS_NAMES = [
    'Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'
];

let model = null;
let currentImageTensor = null;
let isModelLoaded = false;

// ================== INICIALIZACIÓN ==================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Iniciando aplicación...');
    updateStatus('Cargando modelo de IA...', 'loading');
    
    // Inicializar componentes
    initFileUpload();
    loadModel();
});

// ================== CARGA DEL MODELO ==================
async function loadModel() {
    try {
        console.log('📦 Cargando modelo desde model.json...');
        
        // Intentar cargar el modelo (ruta relativa para GitHub Pages)
        model = await tf.loadLayersModel('model.json');
        
        console.log('✅ Modelo cargado correctamente');
        isModelLoaded = true;
        updateStatus('✅ Modelo listo. Sube una imagen de prenda.', 'success');
        
        // Habilitar interfaz
        document.getElementById('predictBtn').disabled = false;
        
    } catch (error) {
        console.error('❌ Error cargando el modelo:', error);
        updateStatus(`❌ Error: ${error.message}. Verifica que model.json esté en la misma carpeta.`, 'error');
    }
}

// ================== MANEJO DE ARCHIVOS ==================
function initFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.querySelector('.upload-area');
    
    fileInput.addEventListener('change', handleImageUpload);
    
    // Soporte para arrastrar y soltar
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f0ebff';
        uploadArea.style.borderColor = '#2575fc';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '#f9f7ff';
        uploadArea.style.borderColor = '#6a11cb';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f9f7ff';
        uploadArea.style.borderColor = '#6a11cb';
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleImageUpload();
        }
    });
}

// ================== PROCESAMIENTO DE IMAGEN ==================
async function handleImageUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    if (!file.type.match('image.*')) {
        alert('⚠️ Por favor, sube un archivo de imagen (JPG, PNG, GIF)');
        return;
    }
    
    updateStatus('Procesando imagen...', 'loading');
    
    const reader = new FileReader();
    
    reader.onload = async function(e) {
        const img = new Image();
        
        img.onload = async function() {
            try {
                // 1. Procesar y normalizar la imagen
                currentImageTensor = processAndNormalizeImage(img);
                
                // 2. Mostrar vista previa
                showImagePreview(currentImageTensor);
                
                // 3. Actualizar interfaz
                document.getElementById('imagePreview').style.display = 'block';
                updateStatus('✅ Imagen procesada. Haz clic en "Identificar Prenda".', 'success');
                
                console.log('🖼️ Imagen procesada correctamente');
                
            } catch (error) {
                console.error('Error procesando imagen:', error);
                updateStatus('❌ Error procesando la imagen', 'error');
            }
        };
        
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

// ================== NORMALIZACIÓN CORRECTA ==================
function processAndNormalizeImage(imgElement) {
    return tf.tidy(() => {
        console.log('🔄 Normalizando imagen...');
        
        // 1. Convertir a tensor
        let tensor = tf.browser.fromPixels(imgElement);
        console.log('   Forma original:', tensor.shape);
        
        // 2. Convertir a escala de grises si es RGB
        if (tensor.shape[2] === 3) {
            tensor = tf.mean(tensor, 2).expandDims(2);
            console.log('   Convertido a escala de grises');
        }
        
        // 3. Redimensionar a 28x28 (EXACTO como el modelo espera)
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        console.log('   Redimensionado a 28x28');
        
        // 4. Invertir colores (Fashion MNIST tiene fondo negro)
        // Esto es CRÍTICO: el dataset original tiene prendas blancas sobre fondo negro
        tensor = tf.sub(1.0, tensor);
        
        // 5. Normalizar valores a [0, 1]
        tensor = tensor.div(255.0);
        
        // 6. Añadir dimensión de batch: [1, 28, 28, 1]
        const finalTensor = tensor.expandDims(0);
        
        // Verificar normalización
        const minVal = finalTensor.min().dataSync()[0];
        const maxVal = finalTensor.max().dataSync()[0];
        console.log(`   ✅ Normalizado. Rango: ${minVal.toFixed(3)} a ${maxVal.toFixed(3)}`);
        
        if (minVal < 0 || maxVal > 1) {
            console.warn('   ⚠️ Atención: valores fuera del rango 0-1');
        }
        
        return finalTensor;
    });
}

// ================== VISTA PREVIA ==================
function showImagePreview(tensor) {
    tf.tidy(() => {
        const canvas = document.getElementById('previewCanvas');
        const ctx = canvas.getContext('2d');
        
        // Crear una versión para visualización (escala de grises)
        const displayTensor = tensor.squeeze().mul(255);
        
        // Dibujar en el canvas (se amplía a 280x280 para mejor visualización)
        const imageData = new ImageData(280, 280);
        
        // Obtener datos del tensor
        const tensorData = displayTensor.dataSync();
        
        // Escalar 10x para visualización
        for (let y = 0; y < 280; y++) {
            for (let x = 0; x < 280; x++) {
                const origX = Math.floor(x / 10);
                const origY = Math.floor(y / 10);
                const idx = origY * 28 + origX;
                const pixelValue = tensorData[idx];
                
                const pos = (y * 280 + x) * 4;
                imageData.data[pos] = pixelValue;     // R
                imageData.data[pos + 1] = pixelValue; // G
                imageData.data[pos + 2] = pixelValue; // B
                imageData.data[pos + 3] = 255;        // Alpha
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    });
}

// ================== PREDICCIÓN (LÓGICA PRINCIPAL) ==================
async function predictImage() {
    if (!isModelLoaded) {
        alert('🔄 El modelo aún se está cargando. Por favor, espera.');
        return;
    }
    
    if (!currentImageTensor) {
        alert('📸 Primero sube una imagen de una prenda.');
        return;
    }
    
    updateStatus('🧠 Analizando imagen con IA...', 'loading');
    
    try {
        console.log('🤖 Realizando predicción...');
        const startTime = performance.now();
        
        // 1. Hacer predicción
        const predictions = model.predict(currentImageTensor);
        const scores = await predictions.data();
        
        // 2. Encontrar la clase con mayor probabilidad
        let maxScore = -1;
        let predictedClass = -1;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                predictedClass = i;
            }
        }
        
        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(0);
        
        // 3. Preparar resultado
        const confidence = (maxScore * 100).toFixed(1);
        const className = CLASS_NAMES[predictedClass];
        
        console.log(`✅ Predicción: ${className} (${confidence}%) en ${inferenceTime}ms`);
        
        // 4. Mostrar ALERTA (como pediste)
        alert(`🎯 RESULTADO:\n\n` +
              `Prenda identificada: ${className}\n` +
              `Confianza: ${confidence}%\n` +
              `Tiempo de análisis: ${inferenceTime}ms\n\n` +
              `(Clase #${predictedClass})`);
        
        // 5. También mostrar en la interfaz
        updateStatus(`✅ Identificado: ${className} (${confidence}% confianza)`, 'success');
        
        // 6. Liberar memoria
        predictions.dispose();
        
    } catch (error) {
        console.error('❌ Error en la predicción:', error);
        updateStatus('❌ Error al analizar la imagen', 'error');
        alert('❌ Ocurrió un error al procesar la imagen. Verifica la consola para más detalles.');
    }
}

// ================== UTILIDADES ==================
function updateStatus(message, type = 'loading') {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.className = `status ${type}`;
    statusElement.style.display = 'block';
    
    console.log(`📢 Estado: ${message}`);
}

// ================== GESTIÓN DE MEMORIA ==================
// Limpiar memoria cuando se cierre la página
window.addEventListener('beforeunload', () => {
    if (currentImageTensor) {
        currentImageTensor.dispose();
    }
    if (model) {
        tf.disposeVariables();
    }
    console.log('🧹 Memoria liberada');
});

// ================== COMPATIBILIDAD GITHUB PAGES ==================
// Solución para problemas de caché en GitHub Pages
if (window.location.hostname.includes('github.io')) {
    console.log('🌐 Detectado GitHub Pages');
    
    // Forzar recarga del modelo si hay error de caché
    window.addEventListener('load', function() {
        const links = document.querySelectorAll('link[rel="stylesheet"], script[src]');
        links.forEach(link => {
            const url = new URL(link.href || link.src, window.location.href);
            url.searchParams.set('v', Date.now());
            if (link.href) link.href = url.href;
            if (link.src) link.src = url.href;
        });
    });
}