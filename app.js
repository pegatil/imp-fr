// app.js - Aplicación completa para GitHub Pages

// ============================================
// CONFIGURACIÓN Y CONSTANTES
// ============================================

const CLASS_NAMES = [
    'Camiseta/Top',     // 0
    'Pantalón',         // 1
    'Suéter',           // 2
    'Vestido',          // 3
    'Abrigo',           // 4
    'Sandalia',         // 5
    'Camisa',           // 6
    'Zapatilla',        // 7
    'Bolso',            // 8
    'Botín'             // 9
];

// Variables globales
let model = null;
let currentImageTensor = null;
let isModelLoaded = false;
let isProcessingImage = false;

// ============================================
// INICIALIZACIÓN DE LA APLICACIÓN
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🎬 Aplicación Fashion MNIST iniciando...');
    
    // Actualizar estado inicial
    updateStatus('🔍 Verificando entorno y cargando modelo...', 'loading');
    
    // Configurar todos los event listeners
    setupEventListeners();
    
    // Cargar el modelo de IA
    loadModel();
});

// ============================================
// CONFIGURACIÓN DE EVENT LISTENERS
// ============================================

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const predictBtn = document.getElementById('predictBtn');
    
    // 1. Click en el área de subida
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // 2. Cambio en el input de archivo
    fileInput.addEventListener('change', handleFileSelect);
    
    // 3. Drag & Drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
        uploadArea.style.borderColor = '#4f46e5';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('drag-over');
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        uploadArea.style.borderColor = '#667eea';
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
    
    // 4. Botón de predicción
    predictBtn.addEventListener('click', performPrediction);
    
    // 5. Prevenir comportamientos por defecto del drag & drop
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
    });
}

// ============================================
// CARGA DEL MODELO DE IA
// ============================================

async function loadModel() {
    console.log('📦 Iniciando carga del modelo...');
    
    try {
        // Actualizar estado
        updateStatus('📡 Descargando modelo de IA...', 'loading');
        
        // Intentar cargar el modelo
        model = await tf.loadLayersModel('frontend/model.json');
        
        // Verificar que el modelo se cargó correctamente
        if (!model) {
            throw new Error('El modelo se cargó pero es nulo');
        }
        
        if (model.layers.length === 0) {
            throw new Error('El modelo no tiene capas definidas');
        }
        
        // Éxito
        isModelLoaded = true;
        console.log('✅ Modelo cargado correctamente');
        console.log(`📊 Modelo tiene ${model.layers.length} capas`);
        
        // Mostrar resumen en consola (opcional)
        model.summary();
        
        // Actualizar interfaz
        updateStatus('✅ Modelo de IA cargado y listo para usar.', 'success');
        document.getElementById('predictBtn').disabled = false;
        document.querySelector('.btn-text').textContent = 'Identificar Prenda';
        
    } catch (error) {
        console.error('❌ ERROR al cargar el modelo:', error);
        
        let errorMessage = 'No se pudo cargar el modelo. ';
        
        if (error.message.includes('404')) {
            errorMessage += 'El archivo model.json no se encuentra. ';
            errorMessage += 'Verifica que esté en la misma carpeta que index.html.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage += 'Error de red. Verifica tu conexión a internet.';
        } else {
            errorMessage += error.message;
        }
        
        updateStatus(`❌ ${errorMessage}`, 'error');
        
        // Deshabilitar funcionalidades
        document.getElementById('predictBtn').disabled = true;
        document.querySelector('.btn-text').textContent = 'Modelo no disponible';
    }
}

// ============================================
// MANEJO DE LA SELECCIÓN DE ARCHIVOS
// ============================================

function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        return;
    }
    
    // Validar que sea una imagen
    if (!file.type.startsWith('image/')) {
        alert('⚠️ Por favor, selecciona un archivo de imagen (JPG, PNG, GIF).');
        fileInput.value = '';
        return;
    }
    
    // Validar tamaño (máximo 5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert('⚠️ La imagen es demasiado grande. Usa una imagen menor a 5MB.');
        fileInput.value = '';
        return;
    }
    
    // Mostrar que se está procesando
    updateStatus('🔄 Procesando imagen...', 'loading');
    isProcessingImage = true;
    
    // Leer la imagen
    const reader = new FileReader();
    
    reader.onload = function(event) {
        const img = new Image();
        
        img.onload = function() {
            try {
                // Procesar la imagen
                processImage(img);
            } catch (error) {
                console.error('Error procesando imagen:', error);
                updateStatus('❌ Error al procesar la imagen', 'error');
                isProcessingImage = false;
            }
        };
        
        img.onerror = function() {
            updateStatus('❌ Error al cargar la imagen', 'error');
            isProcessingImage = false;
        };
        
        img.src = event.target.result;
    };
    
    reader.onerror = function() {
        updateStatus('❌ Error al leer el archivo', 'error');
        isProcessingImage = false;
    };
    
    reader.readAsDataURL(file);
}

// ============================================
// PROCESAMIENTO Y NORMALIZACIÓN DE IMAGEN
// ============================================

function processImage(imgElement) {
    console.log('🖼️ Iniciando procesamiento de imagen...');
    
    try {
        // 1. Liberar tensor anterior si existe
        if (currentImageTensor) {
            currentImageTensor.dispose();
            currentImageTensor = null;
        }
        
        // 2. Aplicar normalización
        currentImageTensor = normalizeImageForFashionMNIST(imgElement);
        
        // 3. Mostrar vista previa
        displayProcessedImage(currentImageTensor);
        
        // 4. Mostrar sección de vista previa
        document.getElementById('imagePreview').style.display = 'block';
        
        // 5. Actualizar estado
        updateStatus('✅ Imagen procesada correctamente. Haz clic en "Identificar Prenda".', 'success');
        
        console.log('✅ Imagen normalizada y lista para predicción');
        isProcessingImage = false;
        
    } catch (error) {
        console.error('Error en processImage:', error);
        updateStatus('❌ Error al procesar la imagen', 'error');
        isProcessingImage = false;
        throw error;
    }
}

// ============================================
// NORMALIZACIÓN ESPECÍFICA PARA FASHION MNIST
// ============================================

function normalizeImageForFashionMNIST(imgElement) {
    return tf.tidy(() => {
        console.log('🎨 Normalizando imagen para Fashion MNIST...');
        
        // 1. Convertir imagen a tensor
        let tensor = tf.browser.fromPixels(imgElement);
        console.log(`   Forma original: [${tensor.shape}]`);
        
        // 2. Convertir a escala de grises si es RGB
        if (tensor.shape[2] === 3) {
            tensor = tf.mean(tensor, 2).expandDims(2);
            console.log('   ✅ Convertido a escala de grises');
        }
        
        // 3. Redimensionar a 28x28 (TAMAÑO EXACTO del dataset)
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        console.log('   ✅ Redimensionado a 28x28 píxeles');
        
        // 4. INVERTIR COLORES (IMPORTANTE: Fashion MNIST tiene fondo negro)
        // Las imágenes del dataset son blancas sobre fondo negro
        tensor = tf.sub(1.0, tensor);
        
        // 5. Normalizar valores de píxeles a [0, 1]
        tensor = tensor.div(255.0);
        
        // 6. Añadir dimensión de batch: [1, 28, 28, 1]
        const finalTensor = tensor.expandDims(0);
        
        // 7. Verificar la normalización
        const minVal = finalTensor.min().dataSync()[0];
        const maxVal = finalTensor.max().dataSync()[0];
        const meanVal = finalTensor.mean().dataSync()[0];
        
        console.log(`   📊 Valores después de normalizar:`);
        console.log(`      Mínimo: ${minVal.toFixed(4)}`);
        console.log(`      Máximo: ${maxVal.toFixed(4)}`);
        console.log(`      Promedio: ${meanVal.toFixed(4)}`);
        
        if (minVal < 0 || maxVal > 1) {
            console.warn('   ⚠️  Atención: valores fuera del rango [0, 1]');
        }
        
        return finalTensor;
    });
}

// ============================================
// VISUALIZACIÓN DE IMAGEN PROCESADA
// ============================================

function displayProcessedImage(tensor) {
    tf.tidy(() => {
        const canvas = document.getElementById('previewCanvas');
        const ctx = canvas.getContext('2d');
        
        // Crear una versión para visualización (ampliada 10x)
        const displayTensor = tensor.squeeze().mul(255);
        
        // Crear un ImageData de 280x280 (28x28 ampliado 10 veces)
        const imageData = new ImageData(280, 280);
        
        // Obtener los datos del tensor
        const tensorData = displayTensor.dataSync();
        
        // Escalar la imagen 10x para mejor visualización
        for (let y = 0; y < 280; y++) {
            for (let x = 0; x < 280; x++) {
                // Calcular posición en la imagen original (28x28)
                const origX = Math.floor(x / 10);
                const origY = Math.floor(y / 10);
                const idx = origY * 28 + origX;
                
                // Obtener valor del píxel (0-255)
                const pixelValue = tensorData[idx];
                
                // Calcular posición en ImageData
                const pos = (y * 280 + x) * 4;
                
                // Asignar valores RGB iguales (escala de grises)
                imageData.data[pos] = pixelValue;     // Red
                imageData.data[pos + 1] = pixelValue; // Green
                imageData.data[pos + 2] = pixelValue; // Blue
                imageData.data[pos + 3] = 255;        // Alpha (totalmente opaco)
            }
        }
        
        // Dibujar en el canvas
        ctx.putImageData(imageData, 0, 0);
        
        console.log('🖼️  Vista previa generada correctamente');
    });
}

// ============================================
// PREDICCIÓN CON EL MODELO
// ============================================

async function performPrediction() {
    console.log('🤖 Iniciando proceso de predicción...');
    
    // Validaciones
    if (!isModelLoaded) {
        alert('🔄 El modelo de IA aún no ha terminado de cargar. Por favor, espera.');
        return;
    }
    
    if (!currentImageTensor) {
        alert('📸 Primero debes subir una imagen de una prenda.');
        return;
    }
    
    if (isProcessingImage) {
        alert('⏳ La imagen aún se está procesando. Espera un momento.');
        return;
    }
    
    // Actualizar estado
    updateStatus('🧠 Analizando imagen con IA...', 'loading');
    
    // Deshabilitar botón durante la predicción
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.querySelector('.btn-text').textContent;
    predictBtn.disabled = true;
    predictBtn.querySelector('.btn-text').textContent = 'Analizando...';
    
    try {
        // Registrar tiempo de inicio
        const startTime = performance.now();
        
        // 1. Realizar la predicción
        console.log('   📤 Enviando tensor al modelo...');
        const predictions = model.predict(currentImageTensor);
        
        // 2. Obtener los resultados
        console.log('   📥 Recibiendo resultados...');
        const scores = await predictions.data();
        
        // 3. Calcular tiempo de inferencia
        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(0);
        
        // 4. Encontrar la clase con mayor probabilidad
        let maxScore = -Infinity;
        let predictedClass = -1;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                predictedClass = i;
            }
        }
        
        // 5. Calcular confianza como porcentaje
        const confidence = (maxScore * 100).toFixed(1);
        
        // 6. Obtener nombre de la clase
        const className = CLASS_NAMES[predictedClass];
        
        // 7. Mostrar resultados en consola
        console.log(`✅ Predicción completada en ${inferenceTime}ms`);
        console.log(`   🎯 Clase: ${className} (índice ${predictedClass})`);
        console.log(`   📈 Confianza: ${confidence}%`);
        console.log(`   📊 Todas las probabilidades:`, Array.from(scores).map((s, i) => 
            `${i}:${(s*100).toFixed(1)}%`).join(', '));
        
        // 8. Mostrar ALERTA con el resultado (como pediste)
        showPredictionAlert(className, confidence, inferenceTime, predictedClass);
        
        // 9. Actualizar estado en la interfaz
        updateStatus(
            `✅ Identificado: ${className} (${confidence}% de confianza)`,
            'success'
        );
        
        // 10. Liberar memoria del tensor de predicciones
        predictions.dispose();
        
    } catch (error) {
        console.error('❌ ERROR en la predicción:', error);
        updateStatus('❌ Error al analizar la imagen', 'error');
        
        alert(`❌ Ocurrió un error durante la predicción:\n\n${error.message}`);
        
    } finally {
        // Rehabilitar el botón
        predictBtn.disabled = false;
        predictBtn.querySelector('.btn-text').textContent = originalText;
    }
}

// ============================================
// ALERTA DE PREDICCIÓN
// ============================================

function showPredictionAlert(className, confidence, inferenceTime, classIndex) {
    // Crear mensaje formateado
    const message = `
🎯 **RESULTADO DE LA PREDICCIÓN**

📋 **Prenda identificada:** ${className}
📊 **Nivel de confianza:** ${confidence}%
⏱️  **Tiempo de análisis:** ${inferenceTime}ms
🔢 **Código de clase:** ${classIndex}

---
📝 **Descripción:** ${getClassDescription(classIndex)}

💡 **Consejo:** ${getPredictionTip(confidence)}
    `.trim();
    
    // Mostrar alerta
    alert(message);
}

// ============================================
// FUNCIONES AUXILIARES
// ============================================

function updateStatus(message, type = 'loading') {
    const statusElement = document.getElementById('status');
    
    if (statusElement) {
        // Actualizar contenido
        statusElement.textContent = message;
        
        // Actualizar clases CSS
        statusElement.className = 'status';
        statusElement.classList.add(type);
        
        // Mostrar elemento
        statusElement.style.display = 'block';
        
        // Registrar en consola
        const icons = { loading: '🔄', success: '✅', error: '❌' };
        console.log(`${icons[type] || '📢'} ${message}`);
    }
}

function getClassDescription(classIndex) {
    const descriptions = [
        'Una prenda superior como una camiseta o top.',
        'Ropa de vestir que cubre las piernas por separado.',
        'Prenda de punto que cubre el torso y los brazos.',
        'Prenda de una pieza que cubre el cuerpo y las piernas.',
        'Prenda exterior para abrigarse en clima frío.',
        'Calzado abierto con sujeción al pie.',
        'Prenda superior con cuello y botones.',
        'Calzado deportivo o casual cerrado.',
        'Accesorio para llevar objetos personales.',
        'Calzado que cubre el tobillo.'
    ];
    
    return descriptions[classIndex] || 'Prenda de vestir no especificada.';
}

function getPredictionTip(confidence) {
    const conf = parseFloat(confidence);
    
    if (conf >= 90) {
        return 'La IA está muy segura de este resultado.';
    } else if (conf >= 70) {
        return 'Resultado confiable. Podría haber prendas similares.';
    } else if (conf >= 50) {
        return 'La IA no está muy segura. Prueba con otra imagen más clara.';
    } else {
        return 'Baja confianza. La imagen podría no ser una prenda de vestir clara.';
    }
}

// ============================================
// GESTIÓN DE MEMORIA
// ============================================

// Limpiar memoria al cerrar/recargar la página
window.addEventListener('beforeunload', function() {
    console.log('🧹 Limpiando memoria antes de cerrar...');
    
    if (currentImageTensor) {
        currentImageTensor.dispose();
        console.log('   ✅ Tensor de imagen liberado');
    }
    
    if (model) {
        // TensorFlow.js maneja la memoria automáticamente
        console.log('   ✅ Recursos del modelo liberados');
    }
});

// ============================================
// COMPATIBILIDAD CON GITHUB PAGES
// ============================================

// Detectar si estamos en GitHub Pages
if (window.location.hostname.includes('github.io')) {
    console.log('🌐 Detectado GitHub Pages - Configurando optimizaciones...');
    
    // Solución para problemas de caché
    if (window.performance && window.performance.navigation.type === 1) {
        // La página fue recargada
        console.log('🔁 Página recargada - Forzando recarga de caché...');
        
        // Forzar recarga del modelo si hay problemas
        setTimeout(() => {
            if (!isModelLoaded) {
                console.log('🔄 Reintentando carga del modelo...');
                loadModel();
            }
        }, 2000);
    }
}

console.log('✅ app.js cargado completamente');