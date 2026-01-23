// script.js
let model = null;
let currentTensor = null;

// Etiquetas de Fashion MNIST
const classNames = [
    'Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'
];

// Cargar modelo
async function loadModel() {
    try {
        alert('🔄 Cargando modelo...');
        model = await tf.loadLayersModel('model/model.json');
        alert('✅ Modelo cargado correctamente');
        
        // Verificar arquitectura del modelo
        model.summary();
        
        // Habilitar interacción
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = '🔍 ¿Qué prenda es?';
        
    } catch (error) {
        console.error('❌ Error cargando el modelo:', error);
        alert('Error cargando el modelo. Verifica la consola para detalles.');
    }
}

// NORMALIZACIÓN COMPLETA de la imagen
function normalizeImage(imageTensor) {
    return tf.tidy(() => {
        // 1. Convertir a escala de grises (si es RGB)
        let gray = imageTensor;
        if (imageTensor.shape[2] === 3) {
            gray = tf.mean(imageTensor, 2).expandDims(2);
        }
        
        // 2. Invertir colores (Fashion MNIST tiene fondo negro)
        // Esto asegura que prendas blancas sobre fondo negro se mantengan
        gray = tf.sub(1.0, gray);
        
        // 3. Normalizar valores a [0, 1] (dividir por 255)
        const normalized = tf.div(gray, 255.0);
        
        // 4. Asegurar que tenga la forma correcta [1, 28, 28, 1]
        const reshaped = normalized.reshape([1, 28, 28, 1]);
        
        console.log('📊 Imagen normalizada. Forma:', reshaped.shape);
        console.log('📊 Valores mínimo/máximo:', 
            reshaped.min().dataSync()[0].toFixed(4), 
            '-', 
            reshaped.max().dataSync()[0].toFixed(4)
        );
        
        return reshaped;
    });
}

// Procesar imagen subida
function processUploadedImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                // Canvas para mostrar original
                const originalCanvas = document.getElementById('originalCanvas');
                const originalCtx = originalCanvas.getContext('2d');
                
                // Limpiar canvas
                originalCtx.fillStyle = 'black';
                originalCtx.fillRect(0, 0, 280, 280);
                
                // Dibujar imagen redimensionada a 28x28 (pero mostramos a 280x280)
                originalCtx.drawImage(img, 0, 0, 280, 280);
                
                // Crear canvas temporal para procesamiento
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28;
                tempCanvas.height = 28;
                const tempCtx = tempCanvas.getContext('2d');
                
                // Dibujar en 28x28 para el modelo
                tempCtx.fillStyle = 'white';
                tempCtx.fillRect(0, 0, 28, 28);
                tempCtx.drawImage(img, 0, 0, 28, 28);
                
                // Obtener tensor de la imagen
                const imageTensor = tf.browser.fromPixels(tempCanvas, 1);
                
                // Aplicar normalización
                const normalizedTensor = normalizeImage(imageTensor);
                
                // Mostrar imagen normalizada
                displayNormalizedImage(normalizedTensor);
                
                // Liberar tensor original
                imageTensor.dispose();
                
                resolve(normalizedTensor);
            };
            img.src = e.target.result;
        };
        
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Mostrar imagen normalizada
function displayNormalizedImage(tensor) {
    tf.tidy(() => {
        // Convertir tensor a valores entre 0-255 para visualización
        const displayTensor = tensor.mul(255).cast('int32');
        
        // Dibujar en canvas
        const canvas = document.getElementById('normalizedCanvas');
        const ctx = canvas.getContext('2d');
        
        tf.browser.toPixels(displayTensor.squeeze(), canvas).then(() => {
            console.log('🖼️ Imagen normalizada mostrada en canvas');
        });
    });
}

// Realizar predicción
async function predictImage(tensor) {
    if (!model) {
        alert('El modelo no está cargado aún');
        return;
    }
    
    try {
        console.log('🤖 Realizando predicción...');
        
        // Hacer predicción
        const prediction = model.predict(tensor);
        const scores = await prediction.data();
        
        // Obtener mejor predicción
        let maxScore = 0;
        let predictedClass = 0;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                predictedClass = i;
            }
        }
        
        // Mostrar resultados
        const confidence = (maxScore * 100).toFixed(2);
        const resultElement = document.getElementById('predictionText');
        const confidenceElement = document.getElementById('confidenceText');
        
        resultElement.innerHTML = `<strong>${classNames[predictedClass]}</strong> (Clase ${predictedClass})`;
        confidenceElement.innerHTML = `Confianza: <strong>${confidence}%</strong>`;
        
        // Efecto visual
        resultElement.style.color = '#667eea';
        resultElement.style.fontSize = '1.5em';
        
        console.log(`🎯 Predicción: ${classNames[predictedClass]} (${confidence}%)`);
        
        // Liberar memoria
        prediction.dispose();
        
    } catch (error) {
        console.error('❌ Error en predicción:', error);
        alert('Error al hacer la predicción');
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', async () => {
    // Cargar modelo al inicio
    await loadModel();
    
    // Manejar subida de archivos
    const fileInput = document.getElementById('fileInput');
    const predictBtn = document.getElementById('predictBtn');
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        // Validar tipo de archivo
        if (!file.type.match('image.*')) {
            alert('Por favor, sube un archivo de imagen');
            return;
        }
        
        // Procesar imagen
        try {
            currentTensor = await processUploadedImage(file);
            predictBtn.disabled = false;
            predictBtn.textContent = '🔍 ¡Clasificar Imagen!';
            
            console.log('✅ Imagen procesada y normalizada');
        } catch (error) {
            console.error('Error procesando imagen:', error);
            alert('Error procesando la imagen');
        }
    });
    
    // Manejar predicción
    predictBtn.addEventListener('click', async () => {
        if (currentTensor) {
            await predictImage(currentTensor);
        }
    });
    
    // Soporte para arrastrar y soltar
    const uploadArea = document.querySelector('.upload-area');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#e0e7ff';
        uploadArea.style.borderColor = '#4f46e5';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '#f8f9ff';
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f8f9ff';
        uploadArea.style.borderColor = '#667eea';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.match('image.*')) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else {
            alert('Por favor, suelta solo archivos de imagen');
        }
    });
});

// Función para manejar memoria (importante para apps largas)
function cleanup() {
    if (currentTensor) {
        currentTensor.dispose();
        currentTensor = null;
    }
    tf.disposeVariables();
}

// Limpiar memoria al cerrar/recargar
window.addEventListener('beforeunload', cleanup);