// --- TAILWIND CONFIGURATION ---
// This must be here for Tailwind to apply to dynamically created elements
tailwind.config = {
    darkMode: "class",
    theme: { 
        extend: { 
            colors: { "primary": "#66aaff" }, 
            fontFamily: { "display": ["Outfit", "Space Grotesk", "sans-serif"] } 
        }
    },
};

// --- HTML TEMPLATE FOR EACH CARD ---
const cardTemplate = (data, cardId) => `
  <div class="flex flex-col gap-6">
    <!-- Search Bar -->
    <div class="search-container">
        <input id="search-input-${cardId}" class="w-full" type="text" placeholder="Search location in Peru..."/>
        <button id="search-btn-${cardId}" class="px-4 rounded-lg">Search</button>
    </div>
    
    <div id="media-container-${cardId}" class="aspect-video w-full rounded-lg shadow-md border border-white/10">
        <div id="map-${cardId}" class="w-full h-full"></div>
    </div>
    
    <div class="flex gap-2">
      <button id="location-btn-${cardId}" class="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-primary/10 text-primary font-semibold hover:bg-primary/20 transition-colors">
        <span class="material-symbols-outlined">my_location</span>
        <span>My Location</span>
      </button>
      <button id="simulate-btn-${cardId}" class="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-blue-500/20 text-blue-300 font-semibold hover:bg-blue-500/30 transition-colors">
        <span class="material-symbols-outlined">movie</span>
        <span>Watch Video</span>
      </button>
    </div>
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm mb-1 text-gray-400">Latitude</label>
        <input id="lat-input-${cardId}" class="w-full rounded-lg p-3 border border-white/20 bg-transparent" type="text" placeholder="0.0000" value="${data.lat || ''}"/>
      </div>
      <div>
        <label class="block text-sm mb-1 text-gray-400">Longitude</label>
        <input id="lon-input-${cardId}" class="w-full rounded-lg p-3 border border-white/20 bg-transparent" type="text" placeholder="0.0000" value="${data.lon || ''}"/>
      </div>
      <div>
        <label class="block text-sm mb-1 text-gray-400">Date</label>
        <input id="date-input-${cardId}" class="w-full rounded-lg p-3 border border-white/20 bg-transparent" type="date" value="${data.fecha}"/>
      </div>
      <div>
        <label class="block text-sm mb-1 text-gray-400">Time</label>
        <input id="time-input-${cardId}" class="w-full rounded-lg p-3 border border-white/20 bg-transparent" type="time" value="${data.hora}"/>
      </div>
    </div>
    <button id="consult-btn-${cardId}" class="w-full py-3 px-4 rounded-lg bg-primary text-white font-bold hover:opacity-90 transition-opacity">Get Data</button>
    <div id="results-container-${cardId}" class="border-t border-white/20 pt-6">
      <h3 class="text-lg font-bold mb-4">Results</h3>
      <div class="space-y-4 text-sm">
        <div class="flex justify-between py-2 border-b border-white/10"><span>City</span><span class="font-semibold text-white">${data.ciudad}</span></div>
        <div class="flex justify-between py-2 border-b border-white/10"><span>Country</span><span class="font-semibold text-white">${data.pais}</span></div>
        <div class="flex justify-between py-2 border-b border-white/10"><span>Forecast</span><span class="font-semibold text-white">${data.pronostico}</span></div>
        <div class="flex justify-between py-2 border-b border-white/10"><span>Precipitation (mm)</span><span class="font-semibold text-white">${data.precipitacion}</span></div>
        <div class="flex justify-between py-2 border-b border-white/10"><span>Relative Humidity</span><span class="font-semibold text-white">${data.humedad}%</span></div>
      </div>
      <div class="mt-6 p-4 bg-primary/10 rounded-lg flex items-center justify-between">
        <div><p class="text-sm">Temperature</p><p class="text-2xl font-bold text-white">${data.temp}°C</p></div>
        <span class="material-symbols-outlined text-4xl text-primary">thermostat</span>
      </div>
    </div>
  </div>
`;

// --- APPLICATION LOGIC ---

function initializeCard(cardId, initialData) {
    const cardContainer = document.getElementById(`card${cardId}`);
    cardContainer.innerHTML = cardTemplate(initialData, cardId);

    const map = L.map(`map-${cardId}`).setView([initialData.lat, initialData.lon], 10);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    let marker = L.marker([initialData.lat, initialData.lon]).addTo(map);
    let isShowingVideo = false;
    
    // --- Card Elements ---
    const consultBtn = document.getElementById(`consult-btn-${cardId}`);
    const locationBtn = document.getElementById(`location-btn-${cardId}`);
    const simulateBtn = document.getElementById(`simulate-btn-${cardId}`);
    const latInput = document.getElementById(`lat-input-${cardId}`);
    const lonInput = document.getElementById(`lon-input-${cardId}`);
    const searchInput = document.getElementById(`search-input-${cardId}`);
    const searchBtn = document.getElementById(`search-btn-${cardId}`);
    const dateInput = document.getElementById(`date-input-${cardId}`);
    const timeInput = document.getElementById(`time-input-${cardId}`);
    const resultsContainer = document.getElementById(`results-container-${cardId}`);

    // --- Function to Update UI with new data ---
    function updateUI(data) {
        resultsContainer.innerHTML = `
            <h3 class="text-lg font-bold mb-4">Results</h3>
            <div class="space-y-4 text-sm">
                <div class="flex justify-between py-2 border-b border-white/10"><span>City</span><span class="font-semibold text-white">${data.ciudad}</span></div>
                <div class="flex justify-between py-2 border-b border-white/10"><span>Country</span><span class="font-semibold text-white">${data.pais}</span></div>
                <div class="flex justify-between py-2 border-b border-white/10"><span>Forecast</span><span class="font-semibold text-white">${data.pronostico}</span></div>
                <div class="flex justify-between py-2 border-b border-white/10"><span>Precipitation (mm)</span><span class="font-semibold text-white">${data.precipitacion}</span></div>
                <div class="flex justify-between py-2 border-b border-white/10"><span>Relative Humidity</span><span class="font-semibold text-white">${data.humedad}%</span></div>
            </div>
            <div class="mt-6 p-4 bg-primary/10 rounded-lg flex items-center justify-between">
                <div><p class="text-sm">Temperature</p><p class="text-2xl font-bold text-white">${data.temp}°C</p></div>
                <span class="material-symbols-outlined text-4xl text-primary">thermostat</span>
            </div>
        `;
    }

    // --- Location Search Function ---
    async function handleSearch() {
        const placeName = searchInput.value.trim();
        if (!placeName) return alert("Please enter a location to search.");
        searchBtn.textContent = "Searching...";
        try {
            const response = await fetch('/api/search_location', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ place_name: placeName })
            });
            if (!response.ok) throw new Error('Location not found');
            const data = await response.json();
            latInput.value = data.latitude.toFixed(6);
            lonInput.value = data.longitude.toFixed(6);
            map.setView([data.latitude, data.longitude], 13);
            if (marker) map.removeLayer(marker);
            marker = L.marker([data.latitude, data.longitude]).addTo(map).bindPopup(data.place_name).openPopup();
        } catch (error) {
            alert(error.message);
        } finally {
            searchBtn.textContent = "Search";
        }
    }
    
    // --- Events ---
    searchBtn.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', e => e.key === 'Enter' && handleSearch());

    consultBtn.addEventListener('click', async () => {
        const lat = parseFloat(latInput.value);
        const lon = parseFloat(lonInput.value);
        const date = dateInput.value;
        const time = timeInput.value;

        if (isNaN(lat) || isNaN(lon) || !date || !time) {
            return alert("Please fill out all fields.");
        }
        consultBtn.textContent = "Loading...";
        consultBtn.disabled = true;
        try {
            const response = await fetch('/api/get_comparison_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ latitude: lat, longitude: lon, date, time })
            });
            const data = await response.json();
            // CORRECTED! Do not call initializeCard, just update the UI.
            updateUI(data);
        } catch (error) {
            alert("Error connecting to the server.");
        } finally {
            consultBtn.textContent = "Get Data";
            consultBtn.disabled = false;
        }
    });

    locationBtn.addEventListener('click', () => {
        navigator.geolocation.getCurrentPosition(position => {
            latInput.value = position.coords.latitude.toFixed(6);
            lonInput.value = position.coords.longitude.toFixed(6);
            consultBtn.click();
        });
    });

    map.on('click', function(e) {
        latInput.value = e.latlng.lat.toFixed(6);
        lonInput.value = e.latlng.lng.toFixed(6);
        if (marker) map.removeLayer(marker);
        marker = L.marker(e.latlng).addTo(map);
    });

    // --- NEW! Logic for the video button ---
    simulateBtn.addEventListener('click', () => {
        const mediaContainer = document.getElementById(`media-container-${cardId}`);
        const staticVideoSrc = '/static/videos/MachuPicchuNoche.mp4'; // <- CHANGE THIS PATH TO YOUR VIDEO!

        if (!isShowingVideo) {
            mediaContainer.innerHTML = `
                <video class="w-full h-full rounded-lg object-cover" autoplay loop muted playsinline>
                    <source src="${staticVideoSrc}" type="video/mp4">
                    Your browser does not support videos.
                </video>`;
            simulateBtn.innerHTML = `<span class="material-symbols-outlined">map</span><span>View Map</span>`;
            isShowingVideo = true;
        } else {
            mediaContainer.innerHTML = ''; // Clear the container
            const newMapDiv = document.createElement('div');
            newMapDiv.id = `map-${cardId}`;
            newMapDiv.className = 'w-full h-full';
            mediaContainer.appendChild(newMapDiv);

            // It's necessary to re-initialize the map
            const lat = parseFloat(latInput.value) || initialData.lat;
            const lon = parseFloat(lonInput.value) || initialData.lon;
            const newMap = L.map(`map-${cardId}`).setView([lat, lon], 10);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(newMap);
            marker = L.marker([lat, lon]).addTo(newMap);

            simulateBtn.innerHTML = `<span class="material-symbols-outlined">movie</span><span>Watch Video</span>`;
            isShowingVideo = false;
        }
    });
}

// --- PAGE INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    const now = new Date().toTimeString().slice(0, 5);

    const initialData1 = { ciudad: "Lima", pais: "Peru", fecha: today, hora: now, pronostico: "...", temp: "...", precipitacion: "...", humedad: "...", lat: -12.0464, lon: -77.0428 };
    const initialData2 = { ciudad: "Arequipa", pais: "Peru", fecha: today, hora: now, pronostico: "...", temp: "...", precipitacion: "...", humedad: "...", lat: -16.4090, lon: -71.5375 };

    initializeCard(1, initialData1);
    initializeCard(2, initialData2);
});

