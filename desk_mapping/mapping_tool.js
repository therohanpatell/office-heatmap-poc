const MappingTool = {
    // --- State Properties ---
    imageWidth: 0,
    imageHeight: 0,
    coordinateData: [],
    isEditMode: false,
    selectedAreaIndex: -1,
    isDrawing: false,
    startX: 0,
    startY: 0,
    currentZoom: 1,
    drawingBox: null,
    deskCounter: 1,
    meetingRoomCounter: 1,

    // --- DOM Element References ---
    elements: {},

    /**
     * Initializes the entire application.
     * Caches DOM elements and sets up initial event listeners.
     */
    init() {
        // Cache frequently accessed DOM elements
        this.elements = {
            floorplanInput: document.getElementById('floorplanInput'),
            csvInput: document.getElementById('csvInput'),
            deskPrefix: document.getElementById('deskPrefix'),
            meetingPrefix: document.getElementById('meetingPrefix'),
            deskPreview: document.getElementById('deskPreview'),
            meetingPreview: document.getElementById('meetingPreview'),
            itemType: document.getElementById('itemType'),
            currentType: document.getElementById('currentType'),
            nextId: document.getElementById('nextId'),
            modeIndicator: document.getElementById('modeIndicator'),
            modeButtonText: document.getElementById('modeButtonText'),
            drawingInfo: document.getElementById('drawingInfo'),
            currentMapping: document.getElementById('currentMapping'),
            zoomLevel: document.getElementById('zoomLevel'),
            floorplanWrapper: document.getElementById('floorplanWrapper'),
            dataTableBody: document.getElementById('dataTableBody'),
        };

        // Bind event listeners
        this.elements.floorplanInput.addEventListener('change', this.handleFloorplanUpload.bind(this));
        this.elements.csvInput.addEventListener('change', this.handleCSVUpload.bind(this));
        this.elements.deskPrefix.addEventListener('input', this.updatePreviews.bind(this));
        this.elements.meetingPrefix.addEventListener('input', this.updatePreviews.bind(this));
        this.elements.itemType.addEventListener('change', this.updateCurrentMapping.bind(this));

        // Initial state setup
        this.updatePreviews();
        this.updateCurrentMapping();
    },

    /**
     * Updates the UI to show the next available ID for both types.
     */
    updatePreviews() {
        const deskPrefix = this.elements.deskPrefix.value || 'D';
        const meetingPrefix = this.elements.meetingPrefix.value || 'MR';

        this.elements.deskPreview.textContent = `Next: ${deskPrefix}-${String(this.deskCounter).padStart(3, '0')}`;
        this.elements.meetingPreview.textContent = `Next: ${meetingPrefix}-${String(this.meetingRoomCounter).padStart(3, '0')}`;

        this.updateCurrentMapping();
    },

    /**
     * Updates the "Currently Mapping" panel based on the selected item type.
     */
    updateCurrentMapping() {
        const itemType = this.elements.itemType.value;
        const deskPrefix = this.elements.deskPrefix.value || 'D';
        const meetingPrefix = this.elements.meetingPrefix.value || 'MR';

        if (itemType === 'desk') {
            this.elements.currentType.textContent = 'Desk';
            this.elements.nextId.textContent = `Next ID: ${deskPrefix}-${String(this.deskCounter).padStart(3, '0')}`;
        } else {
            this.elements.currentType.textContent = 'Meeting Room';
            this.elements.nextId.textContent = `Next ID: ${meetingPrefix}-${String(this.meetingRoomCounter).padStart(3, '0')}`;
        }
    },

    /**
     * Generates the next sequential ID for a given asset type.
     * @param {string} type - 'desk' or 'meeting-room'.
     * @returns {string} The new unique ID.
     */
    generateNextId(type) {
        const deskPrefix = this.elements.deskPrefix.value || 'D';
        const meetingPrefix = this.elements.meetingPrefix.value || 'MR';

        if (type === 'desk') {
            const id = `${deskPrefix}-${String(this.deskCounter).padStart(3, '0')}`;
            this.deskCounter++;
            return id;
        } else {
            const id = `${meetingPrefix}-${String(this.meetingRoomCounter).padStart(3, '0')}`;
            this.meetingRoomCounter++;
            return id;
        }
    },

    /**
     * Scans the existing data to set the ID counters to the next available number.
     */
    updateCountersFromData() {
        const deskPrefix = this.elements.deskPrefix.value || 'D';
        const meetingPrefix = this.elements.meetingPrefix.value || 'MR';

        let maxDeskNum = 0;
        let maxMeetingNum = 0;

        this.coordinateData.forEach(item => {
            const idParts = item.id.split('-');
            if (idParts.length > 1) {
                const num = parseInt(idParts[idParts.length - 1], 10);
                if (!isNaN(num)) {
                    if (item.type === 'desk' && item.id.startsWith(deskPrefix + '-')) {
                        if (num > maxDeskNum) maxDeskNum = num;
                    } else if (item.type === 'meeting-room' && item.id.startsWith(meetingPrefix + '-')) {
                        if (num > maxMeetingNum) maxMeetingNum = num;
                    }
                }
            }
        });

        this.deskCounter = maxDeskNum + 1;
        this.meetingRoomCounter = maxMeetingNum + 1;

        this.updatePreviews();
    },

    /**
     * Handles the floorplan image file upload.
     * @param {Event} event - The file input change event.
     */
    handleFloorplanUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => this.loadFloorplan(e.target.result);
            reader.readAsDataURL(file);
        }
    },

    /**
     * Loads the floorplan image into the DOM.
     * @param {string} imageSrc - The base64 data URL of the image.
     */
    loadFloorplan(imageSrc) {
        this.elements.floorplanWrapper.innerHTML = `
            <div class="floorplan-canvas" id="floorplanCanvas">
                <img src="${imageSrc}" alt="Floorplan">
            </div>
        `;
        const canvas = document.getElementById('floorplanCanvas');
        const img = canvas.querySelector('img');
        img.onload = () => this.setupCanvas(canvas, img);
    },

    /**
     * Sets up the canvas after the image has loaded, getting its dimensions and attaching events.
     * @param {HTMLElement} canvas - The canvas container div.
     * @param {HTMLImageElement} img - The floorplan image element.
     */
    setupCanvas(canvas, img) {
        this.imageWidth = img.naturalWidth;
        this.imageHeight = img.naturalHeight;

        this.resetZoom();

        canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        canvas.addEventListener('mouseleave', this.handleMouseUp.bind(this)); // End drawing if mouse leaves

        this.redrawAreas();
    },

    // --- Mouse Event Handlers for Drawing ---

    handleMouseDown(event) {
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        
        // Convert screen coordinates to image pixel coordinates
        const x = Math.round((event.clientX - rect.left) / canvas.clientWidth * this.imageWidth);
        const y = Math.round((event.clientY - rect.top) / canvas.clientHeight * this.imageHeight);

        // If in edit mode, do nothing on mousedown
        if (this.isEditMode) return;

        // If mapping a desk, just add a point and return
        if (this.elements.itemType.value === 'desk') {
            const id = this.generateNextId('desk');
            this.coordinateData.push({ id, type: 'desk', x, y, width: 20, height: 20 });
            this.updateTable();
            this.redrawAreas();
            this.updateCurrentMapping();
            return;
        }

        // Proceed with drawing a box for meeting rooms
        this.isDrawing = true;
        this.startX = x;
        this.startY = y;

        this.drawingBox = document.createElement('div');
        this.drawingBox.className = 'drawing-box';
        canvas.appendChild(this.drawingBox);
        
        // Initial position for the drawing box
        this.updateDrawingBox(x, y, 0, 0);
    },

    handleMouseMove(event) {
        if (!this.isDrawing) return;

        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const currentX = Math.round((event.clientX - rect.left) / canvas.clientWidth * this.imageWidth);
        const currentY = Math.round((event.clientY - rect.top) / canvas.clientHeight * this.imageHeight);

        const width = Math.abs(currentX - this.startX);
        const height = Math.abs(currentY - this.startY);
        const left = Math.min(this.startX, currentX);
        const top = Math.min(this.startY, currentY);

        this.updateDrawingBox(left, top, width, height);
    },
    
    handleMouseUp(event) {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const endX = Math.round((event.clientX - rect.left) / canvas.clientWidth * this.imageWidth);
        const endY = Math.round((event.clientY - rect.top) / canvas.clientHeight * this.imageHeight);

        if (this.drawingBox) {
            this.drawingBox.remove();
            this.drawingBox = null;
        }

        const width = Math.abs(endX - this.startX);
        const height = Math.abs(endY - this.startY);

        if (width > 10 && height > 10) { // Threshold to prevent tiny boxes
            const left = Math.min(this.startX, endX);
            const top = Math.min(this.startY, endY);
            const type = this.elements.itemType.value;
            const id = this.generateNextId(type);
            
            this.coordinateData.push({ id, type, x: left, y: top, width, height });
            
            this.updateTable();
            this.redrawAreas();
            this.updateCurrentMapping();
        }
    },

    /**
     * Updates the position and size of the temporary drawing box.
     * Coordinates are in image pixels.
     */
    updateDrawingBox(x, y, w, h) {
        if (!this.drawingBox) return;
        // Convert pixel coordinates to percentages for display
        this.drawingBox.style.left = `${(x / this.imageWidth * 100)}%`;
        this.drawingBox.style.top = `${(y / this.imageHeight * 100)}%`;
        this.drawingBox.style.width = `${(w / this.imageWidth * 100)}%`;
        this.drawingBox.style.height = `${(h / this.imageHeight * 100)}%`;
    },

    /**
     * Clears and redraws all mapped areas and their labels on the canvas.
     */
    redrawAreas() {
        const canvas = document.getElementById('floorplanCanvas');
        if (!canvas || this.imageWidth === 0) return;

        // Clear existing areas
        canvas.querySelectorAll('.coordinate-box').forEach(area => area.remove());

        this.coordinateData.forEach((item, index) => {
            const area = document.createElement('div');
            area.className = `coordinate-box ${item.type}`;
            area.style.left = `${(item.x / this.imageWidth * 100)}%`;
            area.style.top = `${(item.y / this.imageHeight * 100)}%`;
            area.style.width = `${(item.width / this.imageWidth * 100)}%`;
            area.style.height = `${(item.height / this.imageHeight * 100)}%`;
            area.title = `${item.id} (${item.type}) - Pixel: ${item.x},${item.y}`;

            const label = document.createElement('div');
            label.textContent = item.id;
            area.appendChild(label);

            if (this.isEditMode) {
                area.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.selectArea(index);
                });
            }

            if (index === this.selectedAreaIndex) {
                area.classList.add('selected');
            }

            canvas.appendChild(area);
        });
    },

    /**
     * Selects an area in edit mode.
     * @param {number} index - The index of the area in the coordinateData array.
     */
    selectArea(index) {
        this.selectedAreaIndex = index;
        this.redrawAreas(); // Redrawing will handle adding the 'selected' class
    },

    /**
     * Toggles between Drawing and Edit modes.
     */
    toggleMode() {
        this.isEditMode = !this.isEditMode;
        if (this.isEditMode) {
            this.elements.modeIndicator.textContent = 'Edit Mode: Click areas to select';
            this.elements.modeButtonText.textContent = 'Switch to Drawing Mode';
            this.elements.drawingInfo.style.display = 'none';
            this.elements.currentMapping.style.display = 'none';
        } else {
            this.elements.modeIndicator.textContent = 'Drawing Mode: Click/drag to map';
            this.elements.modeButtonText.textContent = 'Switch to Edit Mode';
            this.elements.drawingInfo.style.display = 'block';
            this.elements.currentMapping.style.display = 'block';
            this.selectedAreaIndex = -1; // Deselect when leaving edit mode
        }
        this.redrawAreas();
    },

    // --- Zoom Controls ---

    zoomIn() {
        this.currentZoom = Math.min(this.currentZoom * 1.2, 5);
        this.applyZoom();
    },
    zoomOut() {
        this.currentZoom = Math.max(this.currentZoom / 1.2, 0.2);
        this.applyZoom();
    },
    resetZoom() {
        this.currentZoom = 1;
        this.applyZoom();
    },
    applyZoom() {
        const canvas = document.getElementById('floorplanCanvas');
        if (canvas) {
            canvas.style.transform = `scale(${this.currentZoom})`;
            this.elements.zoomLevel.textContent = `${Math.round(this.currentZoom * 100)}%`;
        }
    },

    // --- Data Table and Actions ---

    /**
     * Updates the HTML table with the current coordinate data.
     */
    updateTable() {
        this.elements.dataTableBody.innerHTML = '';
        this.coordinateData.forEach((item, index) => {
            const row = this.elements.dataTableBody.insertRow();
            row.innerHTML = `
                <td>${item.id}</td>
                <td>${item.type}</td>
                <td>${item.x}</td>
                <td>${item.y}</td>
                <td>${item.width}</td>
                <td>${item.height}</td>
                <td><button class="delete-btn" onclick="MappingTool.deleteArea(${index})">×</button></td>
            `;
        });
    },

    deleteArea(index) {
        this.coordinateData.splice(index, 1);
        this.updateTable();
        this.redrawAreas();
        this.selectedAreaIndex = -1;
        this.updateCountersFromData();
    },

    clearAll() {
        if (confirm('Are you sure you want to clear all mapped areas? This cannot be undone.')) {
            this.coordinateData = [];
            this.deskCounter = 1;
            this.meetingRoomCounter = 1;
            this.selectedAreaIndex = -1;
            this.updateTable();
            this.redrawAreas();
            this.updatePreviews();
        }
    },

    // --- File Handling (CSV) ---

    downloadCSV() {
        if (this.coordinateData.length === 0) {
            alert('No data to export. Please map some areas first.');
            return;
        }

        const headers = ['ID', 'Type', 'X_Pixels', 'Y_Pixels', 'Width_Pixels', 'Height_Pixels'];
        const csvContent = [
            headers.join(','),
            ...this.coordinateData.map(item =>
                [item.id, item.type, item.x, item.y, item.width, item.height].join(',')
            )
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'office_coordinates.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    },

    handleCSVUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => this.parseCSV(e.target.result);
            reader.readAsText(file);
        }
    },

    parseCSV(csvText) {
        try {
            const lines = csvText.trim().split(/\r?\n/);
            const headers = lines[0].toLowerCase().split(',').map(h => h.trim());
            
            // Basic validation
            const requiredHeaders = ['id', 'type', 'x_pixels', 'y_pixels', 'width_pixels', 'height_pixels'];
            if (!requiredHeaders.every(h => headers.includes(h))) {
                throw new Error('CSV file is missing required headers.');
            }

            this.coordinateData = [];
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',');
                if (values.length >= 6 && values[0].trim()) {
                    this.coordinateData.push({
                        id: values[0].trim(),
                        type: values[1].trim(),
                        x: parseInt(values[2], 10),
                        y: parseInt(values[3], 10),
                        width: parseInt(values[4], 10),
                        height: parseInt(values[5], 10)
                    });
                }
            }
            
            this.updateTable();
            this.redrawAreas();
            this.updateCountersFromData();
            alert(`Successfully loaded ${this.coordinateData.length} coordinate areas.`);
        } catch (error) {
            alert(`Error parsing CSV: ${error.message}`);
        }
    }
};

// Initialize the application once the DOM is fully loaded.
document.addEventListener('DOMContentLoaded', () => {
    MappingTool.init();
}); 