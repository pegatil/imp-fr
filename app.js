// Variables globales
let model = null;
let imageTensor = null;

// Nombres de las clases (Fashion MNIST)
const classNames = [
    'Camiseta/Top',
    'Pantalón',
    'Suéter',
    'Vestido',
    'Abrigo',
    'Sandalia',
    'Camisa',
    'Zapatilla',
    'Bolso',
    'Bota Tobillera'
];

// Referencias a elementos del DOM
const imageInput = document.getElementById('imageInput');
const selectButton = document.getElementById('selectButton');
const clearButton = document.getElementById('clearButton');
const predictButton = document.getElementById('predictButton');
const uploadArea = document.getElementById('uploadArea');
const imageCanvas = document.getElementById('imageCanvas');
const imagePlaceholder = document.getElementById('imagePlaceholder');
const loadingIndicator = document.getElementById('loadingIndicator');
const results = document.getElementById('results');
const errorMessage = document.getElementById('errorMessage');
const predictedClass = document.getElementById('predictedClass');
const confidenceValue = document.getElementById('confidenceValue');
const probabilitiesList = document.getElementById('probabilitiesList');

// Cargar el modelo cuando se inicie la aplicación
async function loadModel() {
    try {
        console.log('Cargando modelo...');
        // Asegúrate de que la ruta sea correcta para tu modelo en GitHub Pages
        model = await tf.loadLayersModel('/frontend/model.json');
        console.log('Modelo cargado exitosamente');
        
        // Calentamiento del modelo (opcional, pero recomendado)
        const warmupResult = model.predict(tf.zeros([1, 28, 28, 1]));
        warmupResult.dispose();
        
        // Habilitar el botón de predicción si hay una imagen cargada
        if (imageTensor) {
            predictButton.disabled = false;
        }
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        alert('Error al cargar el modelo. Asegúrate de que los archivos del modelo estén en la carpeta "model" y accesibles.');
    }
}

// Función para preprocesar la imagen (similar a tu función de normalización)
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        // Convertir la imagen a tensor
        let tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([28, 28]) // Redimensionar a 28x28
            .mean(2) // Convertir a escala de grises (promedio de los canales RGB)
            .expandDims(2) // Añadir dimensión del canal (28,28) -> (28,28,1)
            .expandDims() // Añadir dimensión del batch -> (1,28,28,1)
            .toFloat(); // Convertir a float
        
        // Normalizar de 0-255 a 0-1 (igual que tu función)
        tensor = tensor.div(255.0);
        
        return tensor;
    });
}

// Función para mostrar la imagen subida en el canvas
function displayImageOnCanvas(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const img = new Image();
            
            img.onload = function() {
                const ctx = imageCanvas.getContext('2d');
                
                // Limpiar el canvas
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                
                // Calcular dimensiones para mantener la relación de aspecto
                const scale = Math.min(
                    imageCanvas.width / img.width,
                    imageCanvas.height / img.height
                );
                
                const width = img.width * scale;
                const height = img.height * scale;
                const x = (imageCanvas.width - width) / 2;
                const y = (imageCanvas.height - height) / 2;
                
                // Dibujar la imagen en el canvas
                ctx.drawImage(img, x, y, width, height);
                
                // Ocultar placeholder y mostrar canvas
                imagePlaceholder.style.display = 'none';
                imageCanvas.style.display = 'block';
                
                // Preprocesar la imagen para el modelo
                imageTensor = preprocessImage(img);
                
                // Habilitar el botón de predicción si el modelo está cargado
                if (model) {
                    predictButton.disabled = false;
                }
                
                resolve();
            };
            
            img.onerror = reject;
            img.src = e.target.result;
        };
        
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Función para realizar la predicción
async function predictImage() {
    if (!model || !imageTensor) {
        alert('Por favor, carga un modelo y una imagen primero.');
        return;
    }
    
    try {
        // Mostrar indicador de carga
        loadingIndicator.style.display = 'block';
        results.style.display = 'none';
        errorMessage.style.display = 'none';
        
        // Realizar la predicción
        const predictions = model.predict(imageTensor);
        
        // Obtener las probabilidades
        const probabilities = await predictions.data();
        predictions.dispose();
        
        // Encontrar la clase con mayor probabilidad
        let maxProb = 0;
        let predictedIndex = 0;
        
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedIndex = i;
            }
        }
        
        // Mostrar resultados
        displayResults(predictedIndex, maxProb, probabilities);
        
    } catch (error) {
        console.error('Error durante la predicción:', error);
        loadingIndicator.style.display = 'none';
        errorMessage.style.display = 'block';
    }
}

// Función para mostrar los resultados
function displayResults(predictedIndex, confidence, probabilities) {
    // Ocultar indicador de carga
    loadingIndicator.style.display = 'none';
    
    // Mostrar resultados
    results.style.display = 'block';
    
    // Actualizar la clase predicha y confianza
    predictedClass.textContent = `${predictedIndex}: ${classNames[predictedIndex]}`;
    confidenceValue.textContent = `${(confidence * 100).toFixed(2)}%`;
    
    // Limpiar lista de probabilidades anteriores
    probabilitiesList.innerHTML = '';
    
    // Crear barras de probabilidad para cada clase
    for (let i = 0; i < classNames.length; i++) {
        const probability = probabilities[i];
        const percentage = (probability * 100).toFixed(2);
        
        // Crear elemento de probabilidad
        const probabilityItem = document.createElement('div');
        probabilityItem.className = 'probability-bar';
        
        probabilityItem.innerHTML = `
            <div class="probability-label">
                <span>${i}: ${classNames[i]}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-container">
                <div class="bar" style="width: ${percentage}%"></div>
            </div>
        `;
        
        probabilitiesList.appendChild(probabilityItem);
    }
}

// Función para limpiar la imagen y resultados
function clearImage() {
    // Limpiar el canvas
    const ctx = imageCanvas.getContext('2d');
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    
    // Mostrar placeholder y ocultar canvas
    imagePlaceholder.style.display = 'block';
    imageCanvas.style.display = 'none';
    
    // Ocultar resultados
    results.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingIndicator.style.display = 'none';
    
    // Deshabilitar botón de predicción
    predictButton.disabled = true;
    
    // Liberar memoria del tensor
    if (imageTensor) {
        imageTensor.dispose();
        imageTensor = null;
    }
}

// Event Listeners
selectButton.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        displayImageOnCanvas(file).catch(error => {
            console.error('Error al cargar la imagen:', error);
            alert('Error al cargar la imagen. Asegúrate de que sea un archivo de imagen válido.');
        });
    }
});

predictButton.addEventListener('click', predictImage);
clearButton.addEventListener('click', clearImage);

// Funcionalidad de arrastrar y soltar
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        displayImageOnCanvas(file).catch(error => {
            console.error('Error al cargar la imagen:', error);
            alert('Error al cargar la imagen. Asegúrate de que sea un archivo de imagen válido.');
        });
    } else {
        alert('Por favor, suelta un archivo de imagen válido.');
    }
});

// Inicializar la aplicación
document.addEventListener('DOMContentLoaded', () => {
    // Cargar el modelo al iniciar
    loadModel();
    
    // También se puede hacer clic en el área de subida para seleccionar imagen
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });
    
    console.log('Aplicación inicializada. Esperando entrada del usuario...');
});

// Manejo de limpieza de memoria de TensorFlow.js
window.addEventListener('beforeunload', () => {
    if (imageTensor) {
        imageTensor.dispose();
    }
    if (model) {
        // TensorFlow.js maneja la limpieza automáticamente en la mayoría de los casos
        tf.disposeVariables();
    }
});