document.addEventListener('DOMContentLoaded', () => {
    const calendar = document.getElementById('calendar');
    const monthYear = document.getElementById('monthYear');
    const modal = document.getElementById('eventModal');
    const modalDate = document.getElementById('modalDate');
    const cancelBtn = document.getElementById('cancelBtn');
    const saveBtn = document.getElementById('saveBtn');

    let selectedDate = null;
    let events = {}; // Objeto para guardar los eventos del servidor
    let currentDate = new Date();

    const months = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ];

    async function fetchEvents() {
        try {
            const response = await fetch('/api/obtener_eventos');
            if (!response.ok) {
                if (response.status === 401) {
                    alert('Tu sesión ha expirado. Por favor, inicia sesión de nuevo.');
                    window.location.href = '/login';
                }
                throw new Error('No se pudieron cargar los eventos.');
            }
            events = await response.json();
            renderCalendar();
        } catch (error) {
            console.error('Error al obtener eventos:', error);
        }
    }

    function renderCalendar() {
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();
        monthYear.textContent = `${months[month]} ${year}`;
        
        const firstDayOfMonth = new Date(year, month, 1);
        const lastDayOfMonth = new Date(year, month + 1, 0);
        const lastDayOfPrevMonth = new Date(year, month, 0);

        calendar.innerHTML = '';

        // Días del mes anterior
        let startDay = firstDayOfMonth.getDay();
        if (startDay === 0) startDay = 7; // Convertir Domingo (0) a 7

        for (let i = startDay - 1; i > 0; i--) {
            const dayNum = lastDayOfPrevMonth.getDate() - i + 1;
            calendar.innerHTML += `<div class="day other-month"><div class="day-number">${dayNum}</div></div>`;
        }

        // Días del mes actual
        for (let i = 1; i <= lastDayOfMonth.getDate(); i++) {
            const fullDate = `${year}-${String(month + 1).padStart(2, '0')}-${String(i).padStart(2, '0')}`;
            const dayEvents = events[fullDate] || [];
            let eventHTML = dayEvents.map(e => `<div class="event" title="${e.desc || ''}">${e.title}</div>`).join("");
            
            calendar.innerHTML += `
              <div class="day" onclick="openModal('${fullDate}')">
                <div class="day-number">${i}</div>
                ${eventHTML}
              </div>`;
        }
    }

    function openModal(dateStr) {
        selectedDate = dateStr;
        const [year, month, day] = dateStr.split('-');
        modalDate.textContent = `${day} de ${months[parseInt(month) - 1]} de ${year}`;
        modal.style.display = 'flex';
        // Limpiar formulario (podríamos añadir lógica para editar eventos aquí)
        document.getElementById('eventTitle').value = '';
        document.getElementById('eventDesc').value = '';
    }

    function closeModal() {
        modal.style.display = 'none';
    }

    async function saveEvent() {
        const title = document.getElementById('eventTitle').value.trim();
        const desc = document.getElementById('eventDesc').value.trim();
        if (!title) return alert("Por favor ingresa un título");

        const [year, month, day] = selectedDate.split('-');
        // Asumimos una hora por defecto si no se especifica, ej. 12:00
        const hora = '12:00'; 

        try {
            const response = await fetch('/api/agendar_evento', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    titulo: title,
                    descripcion: desc,
                    fecha: selectedDate,
                    hora: hora
                })
            });

            if (!response.ok) {
                 const data = await response.json();
                 throw new Error(data.error || 'Error al guardar.');
            }
            
            closeModal();
            fetchEvents(); // Recargar los eventos desde el servidor
        } catch(error) {
            console.error('Error al guardar evento:', error);
            alert(error.message);
        }
    }

    // --- EVENTOS ---
    document.getElementById('prev').addEventListener('click', () => {
        currentDate.setMonth(currentDate.getMonth() - 1);
        renderCalendar();
    });

    document.getElementById('next').addEventListener('click', () => {
        currentDate.setMonth(currentDate.getMonth() + 1);
        renderCalendar();
    });

    cancelBtn.addEventListener('click', closeModal);
    saveBtn.addEventListener('click', saveEvent);

    window.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // Cargar los eventos al iniciar la página
    fetchEvents();
});
