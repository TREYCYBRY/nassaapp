document.addEventListener('DOMContentLoaded', () => {
    // --- ELEMENTOS ---
    const resultsDiv = document.getElementById('results');
    const manualLatInput = document.getElementById('manual-lat');
    const manualLonInput = document.getElementById('manual-lon');
    const manualDateInput = document.getElementById('manual-date');
    const manualTimeInput = document.getElementById('manual-time'); // Referencia al campo de hora

    // --- MAPA LEAFLET ---
    const map = L.map('map').setView([-9.9, -76.2], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);
    let marker = null;
    
    // --- Lógica de clic en el mapa (CORREGIDA) ---
    map.on('click', function(e) {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        manualLatInput.value = lat.toFixed(5);
        manualLonInput.value = lon.toFixed(5);
        if (marker) map.removeLayer(marker);
        marker = L.marker([lat, lon]).addTo(map);
    });

    // --- FUNCIONES ---
    function updateUI(data) {
        const { latitude, longitude, datetime, departamento, pais, prediccion_modelo, temperatura_real } = data;
        if (marker) map.removeLayer(marker);
        map.setView([latitude, longitude], 12);
        marker = L.marker([latitude, longitude]).addTo(map)
            .bindPopup(`<b>${departamento}</b><br>Predicción: ${prediccion_modelo}`)
            .openPopup();
        // ... dentro de la función updateUI ...
        resultsDiv.innerHTML = `
            <h2>Resultados para ${departamento}, ${pais}</h2>
            <div class="result-item">Fecha y Hora: <span>${datetime}</span></div>
            
            <div class="result-item">Pronóstico del Modelo: <span>${prediccion_modelo}</span></div> 
            
            <div class="result-item">Temperatura Real (Histórica): <span>${temperatura_real}</span></div>
        `;
    }

    async function fetchData(lat, lon, date, time) {
        resultsDiv.innerHTML = `<p>Consultando datos...</p>`;
        resultsDiv.style.display = 'block';

        try {
            const response = await fetch('/api/get_location_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // Aseguramos que 'time' se envía correctamente
                body: JSON.stringify({ latitude: lat, longitude: lon, date: date, time: time }),
            });
            const data = await response.json();
            data.latitude = lat;
            data.longitude = lon;
            data.datetime = `${date} ${time}`;
            updateUI(data);
        } catch (error) {
            resultsDiv.innerHTML = `<p>Error: No se pudo conectar con el servidor. ¿Está 'app.py' corriendo?</p>`;
        }
    }

    // --- EVENTOS ---
    document.getElementById('get-location-btn').addEventListener('click', () => {
        if (!navigator.geolocation) return alert('Geolocalización no soportada.');
        navigator.geolocation.getCurrentPosition(pos => {
            const { latitude, longitude } = pos.coords;
            const date = manualDateInput.value;
            const time = manualTimeInput.value; // Obtenemos la hora actual del input
            manualLatInput.value = latitude.toFixed(5);
            manualLonInput.value = longitude.toFixed(5);
            fetchData(latitude, longitude, date, time);
        }, err => alert(`Error de geolocalización: ${err.message}`));
    });

    document.getElementById('get-manual-btn').addEventListener('click', () => {
        const lat = parseFloat(manualLatInput.value);
        const lon = parseFloat(manualLonInput.value);
        const date = manualDateInput.value;
        const time = manualTimeInput.value; // Obtenemos la hora del input
        if (isNaN(lat) || isNaN(lon) || !date || !time) {
            return alert('Por favor, ingresa latitud, longitud, fecha y hora válidas.');
        }
        fetchData(lat, lon, date, time);
    });

    // Pone la fecha de hoy por defecto
    manualDateInput.value = new Date().toISOString().split('T')[0];
});