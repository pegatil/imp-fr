// app.js - VERSIÓN CON DIAGNÓSTICO COMPLETO

// ================== CONFIGURACIÓN ==================
const CLASS_NAMES = [
    'Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'
];

let model = null;
let currentImageTensor = null;

// ================== FUNCIÓN PRINCIPAL DE CARGA ==================
async function initializeApp() {
    console.log('🚀 Inicializando aplicación...');
    
    try {
        // 1. Mostrar estado inmediatamente
        updateStatus('🔍 Verificando archivos del modelo...', 'loading');
        
        // 2. Verificar que model.json existe ANTES de cargarlo
        await checkModelFiles();
        
        // 3. Cargar el modelo
        await loadModelWithRetry();
        
        // 4. Configurar interfaz
        setupFileUpload();
        
    } catch (error) {
        console.error('❌ Error crítico en inicialización:', error);
        updateStatus(`❌ ${error.message}`, 'error');
    }
}

// ================== VERIFICACIÓN DE ARCHIVOS ==================
async function checkModelFiles() {
    console.log('📁 Verificando existencia de model.json...');
    
    try {
        // Intentar acceder al archivo model.json
        const response = await fetch('model.json', { method: 'HEAD' });
        
        if (!response.ok) {
            throw new Error('Archivo model.json NO encontrado');
        }
        
        console.log('✅ model.json encontrado');
        
        // Verificar tamaño del archivo
        const sizeResponse = await fetch('model.json');
        const text = await sizeResponse.text();
        
        if (text.length < 1000) {
            throw new Error('model.json parece estar vacío o corrupto');
        }
        
        console.log(`✅ model.json tiene ${text.length} bytes`);
        
        // Verificar que sea JSON válido
        try {
            JSON.parse(text);
            console.log('✅ model.json es JSON válido');
        } catch (e) {
            throw new Error('model.json no es un JSON válido');
        }
        
    } catch (error) {
        console.error('❌ Error verificando archivos:', error);
        throw new Error(`Verifica que model.json esté en la misma carpeta que index.html. Error: ${error.message}`);
    }
}

// ================== CARGA DEL MODELO CON REINTENTOS ==================
async function loadModelWithRetry(maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            updateStatus(`Cargando modelo (intento ${attempt}/${maxRetries})...`, 'loading');
            console.log(`🔄 Intento ${attempt} de cargar el modelo...`);
            
            // Cargar el modelo desde la RUTA CORRECTA
            model = await tf.loadLayersModel('frontend/model.json');
            
            // Verificar que el modelo se cargó correctamente
            if (!model || model.layers.length === 0) {
                throw new Error('Modelo cargado pero sin capas');
            }
            
            console.log(`✅ Modelo cargado exitosamente en el intento ${attempt}`);
            console.log(`📊 Arquitectura del modelo: ${model.layers.length} capas`);
            
            // Mostrar resumen del modelo en consola
            model.summary();
            
            updateStatus('✅ Modelo listo. Sube una imagen de prenda.', 'success');
            document.getElementById('predictBtn').disabled = false;
            
            return; // Éxito, salir de la función
            
        } catch (error) {
            console.error(`❌ Error en intento ${attempt}:`, error);
            
            if (attempt === maxRetries) {
                throw new Error(`No se pudo cargar el modelo después de ${maxRetries} intentos. Error: ${error.message}`);
            }
            
            // Esperar antes de reintentar (con backoff exponencial)
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
    }
}

// ================== SETUP DE LA INTERFAZ ==================
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.querySelector('.upload-area');
    const predictBtn = document.getElementById('predictBtn');
    
    // Configurar evento de cambio de archivo
    fileInput.addEventListener('change', handleImageSelection);
    
    // Configurar área de drop
    uploadArea.addEventListener('click', () => fileInput.click());
    
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
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleImageSelection();
        }
    });
    
    // Configurar botón de predicción
    predictBtn.addEventListener('click', predictImage);
}

// ================== MANEJO DE IMAGEN SUBIDA ==================
async function handleImageSelection() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('⚠️ Por favor, sube un archivo de imagen (JPG, PNG, GIF)');
        return;
    }
    
    updateStatus('Procesando imagen...', 'loading');
    
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const img = new Image();
        
        img.onload = function() {
            try {
                // Procesar y normalizar la imagen
                currentImageTensor = processAndNormalizeImage(img);
                
                // Mostrar vista previa
                showImagePreview(currentImageTensor);
                
                // Mostrar sección de vista previa
                document.getElementById('imagePreview').style.display = 'block';
                
                updateStatus('✅ Imagen procesada. Haz clic en "Identificar Prenda".', 'success');
                
                console.log('🖼️ Imagen procesada y normalizada');
                
            } catch (error) {
                console.error('Error procesando imagen:', error);
                updateStatus('❌ Error procesando la imagen', 'error');
            }
        };
        
        img.onerror = function() {
            updateStatus('❌ Error al cargar la imagen', 'error');
        };
        
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

// ================== NORMALIZACIÓN DE IMAGEN ==================
function processAndNormalizeImage(imgElement) {
    return tf.tidy(() => {
        console.log('🔄 Normalizando imagen para Fashion MNIST...');
        
        // 1. Convertir imagen a tensor
        let tensor = tf.browser.fromPixels(imgElement);
        console.log('   Forma original:', tensor.shape);
        
        // 2. Convertir a escala de grises si es RGB
        if (tensor.shape[2] === 3) {
            tensor = tf.mean(tensor, 2).expandDims(2);
            console.log('   Convertido a escala de grises');
        }
        
        // 3. Redimensionar a 28x28 (EXACTO como el entrenamiento)
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        console.log('   Redimensionado a 28x28');
        
        // 4. Invertir colores (IMPORTANTE: Fashion MNIST usa fondo negro)
        tensor = tf.sub(1.0, tensor);
        
        // 5. Normalizar a [0, 1]
        tensor = tensor.div(255.0);
        
        // 6. Añadir dimensión de batch: [1, 28, 28, 1]
        const finalTensor = tensor.expandDims(0);
        
        // Verificar normalización
        const minVal = finalTensor.min().dataSync()[0];
        const maxVal = finalTensor.max().dataSync()[0];
        console.log(`   ✅ Normalizado. Rango: ${minVal.toFixed(3)} a ${maxVal.toFixed(3)}`);
        
        return finalTensor;
    });
}

// ================== VISTA PREVIA ==================
function showImagePreview(tensor) {
    tf.tidy(() => {
        const canvas = document.getElementById('previewCanvas');
        const ctx = canvas.getContext('2d');
        
        // Crear versión para visualización
        const displayTensor = tensor.squeeze().mul(255);
        
        // Crear ImageData
        const imageData = new ImageData(280, 280);
        
        // Obtener datos del tensor
        const tensorData = displayTensor.dataSync();
        
        // Escalar 10x para mejor visualización
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

// ================== PREDICCIÓN ==================
async function predictImage() {
    if (!model) {
        alert('🔄 El modelo no está cargado. Espera a que termine la carga.');
        return;
    }
    
    if (!currentImageTensor) {
        alert('📸 Primero sube una imagen de una prenda.');
        return;
    }
    
    updateStatus('🧠 Analizando imagen...', 'loading');
    
    try {
        console.log('🤖 Realizando predicción...');
        const startTime = performance.now();
        
        // Hacer predicción
        const predictions = model.predict(currentImageTensor);
        const scores = await predictions.data();
        
        // Encontrar mejor predicción
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
        
        // Preparar resultado
        const confidence = (maxScore * 100).toFixed(1);
        const className = CLASS_NAMES[predictedClass];
        
        console.log(`🎯 Predicción: ${className} (${confidence}%) en ${inferenceTime}ms`);
        
        // Mostrar alerta
        alert(`🎯 RESULTADO:\n\n` +
              `Prenda identificada: ${className}\n` +
              `Confianza: ${confidence}%\n` +
              `Tiempo: ${inferenceTime}ms\n\n` +
              `(Clase #${predictedClass})`);
        
        updateStatus(`✅ Identificado: ${className} (${confidence}%)`, 'success');
        
        // Liberar memoria
        predictions.dispose();
        
    } catch (error) {
        console.error('❌ Error en predicción:', error);
        updateStatus('❌ Error al analizar la imagen', 'error');
        alert(`❌ Error: ${error.message}`);
    }
}

// ================== UTILIDADES ==================
function updateStatus(message, type = 'loading') {
    const statusElement = document.getElementById('status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
        statusElement.style.display = 'block';
    }
    console.log(`📢 ${message}`);
}

// ================== INICIALIZACIÓN ==================
// Iniciar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', initializeApp);

// Manejar recarga de caché para GitHub Pages
if (window.location.hostname.includes('github.io')) {
    window.addEventListener('pageshow', function(event) {
        if (event.persisted) {
            location.reload();
        }
    });
}