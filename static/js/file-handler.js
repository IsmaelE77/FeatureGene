/**
 * File Handler for CSV upload and processing
 * Handles drag-and-drop, file validation, and CSV parsing
 */

class FileHandler {
    constructor(app) {
        this.app = app;
        this.setupFileUpload();
    }

    setupFileUpload() {
        const fileInput = document.getElementById('csvFile');
        const uploadArea = document.getElementById('fileUploadArea');
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        if (uploadArea) {
            // Make upload area clickable
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
            
            this.setupDragAndDrop(uploadArea);
        }
    }

    setupDragAndDrop(uploadArea) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.highlight(uploadArea), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => this.unhighlight(uploadArea), false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(element) {
        element.classList.add('dragover');
    }

    unhighlight(element) {
        element.classList.remove('dragover');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        try {
            this.app.updateStatus('Reading file...', 'loading');
            
            const csvData = await this.readFileAsText(file);
            
            // Validate CSV content
            if (!this.validateCSVContent(csvData)) {
                this.app.showToast('Invalid CSV format. Please check your file.', 'error');
                return;
            }
            
            // Process successful upload
            this.app.onFileUploaded(file, csvData);
            
        } catch (error) {
            console.error('File processing error:', error);
            this.app.showToast('Error reading file: ' + error.message, 'error');
            this.app.updateStatus('❌ Error reading file: ' + error.message, 'error');
        }
    }

    validateFile(file) {
        // Check file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.app.showToast('Please select a CSV file.', 'error');
            return false;
        }

        // Check file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.app.showToast('File size too large. Maximum size is 10MB.', 'error');
            return false;
        }

        // Check if file is empty
        if (file.size === 0) {
            this.app.showToast('File is empty.', 'error');
            return false;
        }

        return true;
    }

    validateCSVContent(csvData) {
        const lines = csvData.split('\n').filter(line => line.trim());
        
        if (lines.length < 2) {
            this.app.showToast('CSV file must have at least 2 rows (header + data).', 'error');
            return false;
        }

        const headers = lines[0].split(',');
        if (headers.length < 2) {
            this.app.showToast('CSV file must have at least 2 columns.', 'error');
            return false;
        }

        return true;
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                resolve(e.target.result);
            };
            
            reader.onerror = (e) => {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsText(file);
        });
    }

    // Utility method to get file info
    getFileInfo(file) {
        return {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: new Date(file.lastModified)
        };
    }

    // Method to clear current file
    clearFile() {
        const fileInput = document.getElementById('csvFile');
        if (fileInput) {
            fileInput.value = '';
        }
        
        this.app.state.csvData = '';
        this.app.state.currentFile = null;
        
        // Hide column selection
        const columnSelection = document.getElementById('columnSelection');
        if (columnSelection) {
            columnSelection.style.display = 'none';
        }
        
        this.app.updateStatus('No file selected. Please upload a CSV file.');
    }

}
