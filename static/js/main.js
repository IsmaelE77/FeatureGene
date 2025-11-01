/**
 * Main JavaScript file for Genetic Algorithm Feature Selection Application
 * Handles application initialization, theme management, and global state
 */

class GAApplication {
    constructor() {
        this.state = {
            csvData: '',
            currentFile: null,
            isProcessing: false,
            theme: localStorage.getItem('ga-theme') || 'light',
            results: null,
            lastTransformedCsv: null
        };
        
        this.init();
    }

    formatFeatureEngineering(meta) {
        if (!meta) return '';
        const parts = [];
        if (typeof meta.originalFeatureCount === 'number' && typeof meta.transformedFeatureCount === 'number') {
            parts.push(`\nFeature engineering: ${meta.originalFeatureCount} ➜ ${meta.transformedFeatureCount}`);
        }
        if (meta.targetEncoding && meta.targetEncoding.encoded) {
            const classes = (meta.targetEncoding.classes || []).join(', ');
            parts.push(`\nTarget encoding applied. Classes: [${classes}]`);
        }
        const exp = Array.isArray(meta.categoricalExpansions) ? meta.categoricalExpansions : [];
        if (exp.length > 0) {
            const lines = exp.map(e => `${e.column} → +${(e.createdColumns||[]).length} cols`);
            parts.push(`\nOne-Hot expansions: ${lines.join('; ')}`);
        }
        return parts.length ? `\n${parts.join('')}` : '';
    }

    downloadTransformed() {
        const csv = this.state.lastTransformedCsv;
        if (!csv) {
            this.showToast('No transformed dataset available yet. Run an analysis first.', 'warning');
            return;
        }
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'dataset_transformed.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showToast('Transformed dataset downloaded.', 'success');
    }

    async runSelectKBest() {
        if (!this.state.csvData) {
            this.showToast('Please upload a CSV file first!', 'error');
            return;
        }

        const targetColumn = document.getElementById('targetColumn')?.value;
        if (!targetColumn && targetColumn !== '0') {
            this.showToast('Please specify the target column!', 'error');
            return;
        }

        this.setProcessingState(true);
        
        try {
            const result = await this.apiClient.runSelectKBest();
            this.handleSelectKBestResult(result);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.setProcessingState(false);
        }
    }

    init() {
        this.setupTheme();
        this.setupEventListeners();
        this.initializeComponents();
        this.setupDatasetSelector();
        this.updateStatus('Application initialized. Ready to upload dataset.');
    }

    setupTheme() {
        if (this.state.theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.checked = this.state.theme === 'dark';
        }
    }

    setupEventListeners() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('change', (e) => {
                this.toggleTheme(e.target.checked ? 'dark' : 'light');
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Window events
        window.addEventListener('beforeunload', () => {
            this.saveState();
        });
    }

    initializeComponents() {
        // Initialize file handler
        this.fileHandler = new FileHandler(this);

        // Initialize API client
        this.apiClient = new APIClient(this);

        // Initialize chart manager
        this.chartManager = new ChartManager(this);

        // Initialize UI components
        this.uiComponents = new UIComponents(this);
    }

      async setupDatasetSelector() {
        const select = document.getElementById('datasetSelect');
        const btn = document.getElementById('loadDatasetBtn');
        if (!select || !btn) return;
        try {
            const datasets = await this.apiClient.listDatasets();
            while (select.firstChild) select.removeChild(select.firstChild);
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = '-- Select dataset --';
            select.appendChild(placeholder);
            datasets.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        } catch (e) {
            this.showToast('Failed to load dataset list', 'error');
        }
        btn.addEventListener('click', () => this.loadSelectedDataset());
    }

    async loadSelectedDataset() {
        const select = document.getElementById('datasetSelect');
        if (!select || !select.value) {
            this.showToast('Please select a dataset.', 'error');
            return;
        }
        this.setProcessingState(true);
        try {
            const result = await this.apiClient.getDataset(select.value);
            const csvData = result.csvData || '';
            const size = new Blob([csvData]).size;
            const file = { name: result.name || select.value, size };
            this.onFileUploaded(file, csvData);
        } catch (e) {
            this.handleError(e);
        } finally {
            this.setProcessingState(false);
        }
    }

    toggleTheme(theme) {
        this.state.theme = theme;
        
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        
        localStorage.setItem('ga-theme', theme);
        
        // Update chart colors if chart exists
        if (this.chartManager && this.chartManager.currentChart) {
            this.chartManager.updateTheme(theme);
        }
    }

    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + Enter: Run comparison
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            this.runComparison();
        }

        // Ctrl/Cmd + U: Upload file
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            document.getElementById('csvFile').click();
        }

        // Escape: Clear status
        if (e.key === 'Escape') {
            this.clearStatus();
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('status');
        if (!statusElement) return;

        // Clear previous classes
        statusElement.className = 'status-display';
        
        // Add type-specific class (note: shimmer will be applied to child span only)
        if (type === 'loading') {
            statusElement.classList.add('status-loading');
        } else if (type === 'success') {
            statusElement.classList.add('status-success');
        } else if (type === 'error') {
            statusElement.classList.add('status-error');
        }

        // Update content
        const msg = String(message || '');
        if (type === 'loading' || msg.toLocaleLowerCase().includes('running')) {
            // Wrap text in a span so shimmer affects only text, not the container/border
            while (statusElement.firstChild) statusElement.removeChild(statusElement.firstChild);
            const span = document.createElement('span');
            span.className = 'loading-text-shimmer';
            span.textContent = message;
            statusElement.appendChild(span);
        } else {
            // Plain text for non-loading states; ensure no shimmer class on container
            statusElement.textContent = message;
        }

        // Keep shimmer overlay text in sync (preserve base text color)
        // if (type === 'loading') {
        //     statusElement.setAttribute('data-text', message);
        // } else {
        //     statusElement.removeAttribute('data-text');
        //     statusElement.classList.remove('loading-text-shimmer');
        // }
        
        // Scroll to status if it's not visible
        statusElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    clearStatus() {
        this.updateStatus('Ready for next operation.');
    }

    copyResults() {
        const statusElement = document.getElementById('status');
        if (!statusElement) return;
        navigator.clipboard.writeText(statusElement.textContent);
        this.showToast('Results copied to clipboard', 'success');
    }

    setProcessingState(isProcessing) {
        this.state.isProcessing = isProcessing;
        
        // Update UI elements
        const buttons = document.querySelectorAll('.btn');
        buttons.forEach(btn => {
            btn.disabled = isProcessing;
        });

        // Update status
        if (isProcessing) {
            this.updateStatus('Processing... Please wait.', 'loading');
        }
    }

    showToast(message, type = 'info', duration = 3000) {
        if (this.uiComponents) {
            this.uiComponents.showToast(message, type, duration);
        }
    }

    saveState() {
        // Save current state to localStorage
        const stateToSave = {
            theme: this.state.theme,
            parameters: this.getCurrentParameters()
        };
        localStorage.setItem('ga-app-state', JSON.stringify(stateToSave));
    }

    loadState() {
        const savedState = localStorage.getItem('ga-app-state');
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                if (state.parameters) {
                    this.setParameters(state.parameters);
                }
            } catch (e) {
                console.warn('Failed to load saved state:', e);
            }
        }
    }

    getCurrentParameters() {
        return {
            popSize: document.getElementById('popSize')?.value || '30',
            crossRate: document.getElementById('crossRate')?.value || '0.7',
            mutRate: document.getElementById('mutRate')?.value || '0.1',
            maxGen: document.getElementById('maxGen')?.value || '20',
            convergenceThreshold: document.getElementById('convergenceThreshold')?.value || '',
            vtThreshold: document.getElementById('vtThreshold')?.value || '0.0',
            idColumn: document.getElementById('idColumn')?.value || '',
            targetColumn: document.getElementById('targetColumn')?.value || ''
        };
    }

    setParameters(parameters) {
        Object.keys(parameters).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.value = parameters[key];
            }
        });
    }

    // Main execution methods
    async runGA() {
        if (!this.state.csvData) {
            this.showToast('Please upload a CSV file first!', 'error');
            return;
        }

        const targetColumn = document.getElementById('targetColumn')?.value;
        if (!targetColumn && targetColumn !== '0') {
            this.showToast('Please specify the target column!', 'error');
            return;
        }

        this.setProcessingState(true);
        
        try {
            const result = await this.apiClient.runGA();
            this.handleGAResult(result);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.setProcessingState(false);
        }
    }

    async runVarianceThreshold() {
        if (!this.state.csvData) {
            this.showToast('Please upload a CSV file first!', 'error');
            return;
        }

        const targetColumn = document.getElementById('targetColumn')?.value;
        if (!targetColumn && targetColumn !== '0') {
            this.showToast('Please specify the target column!', 'error');
            return;
        }

        this.setProcessingState(true);
        
        try {
            const result = await this.apiClient.runVarianceThreshold();
            this.handleVarianceThresholdResult(result);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.setProcessingState(false);
        }
    }

    async runComparison() {
        if (!this.state.csvData) {
            this.showToast('Please upload a CSV file first!', 'error');
            return;
        }

        const targetColumn = document.getElementById('targetColumn')?.value;
        if (!targetColumn && targetColumn !== '0') {
            this.showToast('Please specify the target column!', 'error');
            return;
        }

        this.setProcessingState(true);
        
        try {
            const result = await this.apiClient.runComparison();
            this.handleComparisonResult(result);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.setProcessingState(false);
        }
    }

    handleGAResult(data) {
        this.state.results = { type: 'ga', data };
        this.state.lastTransformedCsv = data.transformedCsv || null;
        const feSummary = this.formatFeatureEngineering(data.featureEngineering);
        const selectedCount = data.selectedFeatures.length;
        const idInfo = data.idColumn ? `ID Column: ${data.idColumn}\n` : `No ID column\n`;
        const convergenceInfo = data.converged ? 
            `Converged at generation ${data.generations}` : 
            `Completed ${data.generations} generations (max reached)`;
        
        const statusMessage = 
            `✅ GA Complete!\n${idInfo}Target Column: ${data.target}\n` +
            `${convergenceInfo}\n` +
            `Best fitness: ${data.bestFitness}\n` +
            `Exec Time: ${data.execTimeSeconds}s\n` +
            `Selected features: ${selectedCount} out of ${data.featuresCount}\n` +
            `Features: ${data.selectedFeatures.join(', ')}\n` +
            `Total features: ${data.featuresCount}, Rows: ${data.rows}` +
            `${feSummary}`;

        this.updateStatus(statusMessage, 'success');
        
        // Update chart
        if (this.chartManager) {
            this.chartManager.renderGAChart(data);
        }

        this.showToast('Genetic Algorithm completed successfully!', 'success');
    }

    handleVarianceThresholdResult(data) {
        this.state.results = { type: 'variance', data };
        this.state.lastTransformedCsv = data.transformedCsv || null;
        const feSummary = this.formatFeatureEngineering(data.featureEngineering);
        const idInfo = data.idColumn ? `ID Column: ${data.idColumn}\n` : `No ID column\n`;
        const statusMessage = 
            `✅ VarianceThreshold Complete!\n${idInfo}Target Column: ${data.target}\n` +
            `Threshold Used: ${data.thresholdUsed}\n` +
            `Accuracy: ${data.accuracy}\n` +
            `Exec Time: ${data.execTimeSeconds}s\n` +
            `Selected features: ${data.numFeaturesSelected} out of ${data.numFeaturesTotal}\n` +
            `Selected: ${data.selectedFeatures.join(', ')}\n` +
            `Removed: ${data.removedFeatures.join(', ')}\n` +
            `Total features: ${data.numFeaturesTotal}, Rows: ${data.rows}` +
            `${feSummary}`;

        this.updateStatus(statusMessage, 'success');
        
        // Update chart
        if (this.chartManager) {
            this.chartManager.renderVarianceThresholdChart(data);
        }

        this.showToast('Variance Threshold completed successfully!', 'success');
    }

    handleSelectKBestResult(data) {
        this.state.results = { type: 'selectkbest', data };
        this.state.lastTransformedCsv = data.transformedCsv || null;
        const feSummary = this.formatFeatureEngineering(data.featureEngineering);
        const idInfo = data.idColumn ? `ID Column: ${data.idColumn}\n` : `No ID column\n`;
        const statusMessage = 
            `✅ SelectKBest Complete!\n${idInfo}Target Column: ${data.target}\n` +
            `Score Func: ${data.scoreFuncUsed}\n` +
            `K Used: ${data.kUsed}\n` +
            `Accuracy: ${data.accuracy}\n` +
            `Exec Time: ${data.execTimeSeconds}s\n` +
            `Selected features: ${data.numFeaturesSelected} out of ${data.numFeaturesTotal}\n` +
            `Selected: ${data.selectedFeatures.join(', ')}\n` +
            `Removed: ${data.removedFeatures.join(', ')}\n` +
            `Total features: ${data.numFeaturesTotal}, Rows: ${data.rows}` +
            `${feSummary}`;

        this.updateStatus(statusMessage, 'success');
        
        if (this.chartManager) {
            this.chartManager.renderSelectKBestChart(data);
        }
        
        this.showToast('SelectKBest completed successfully!', 'success');
    }

    handleComparisonResult(data) {
        this.state.results = { type: 'comparison', data };
        this.state.lastTransformedCsv = (data.ga && data.ga.transformedCsv) ? data.ga.transformedCsv : null;
        const feSummary = this.formatFeatureEngineering(data.dataset && data.dataset.featureEngineering);
        const idInfo = data.dataset.idColumn ? `ID Column: ${data.dataset.idColumn}\n` : `No ID column\n`;
        const gaConvergence = data.ga.converged ? 
            `Converged at generation ${data.ga.generations}` : 
            `Completed ${data.ga.generations} generations (max reached)`;

        const scores = [
            { name: 'Genetic Algorithm', value: data.ga.accuracy },
            { name: 'VarianceThreshold', value: data.varianceThreshold.accuracy },
            { name: 'SelectKBest', value: data.selectKBest.accuracy }
        ];
        const top = scores.reduce((a, b) => (b.value > a.value ? b : a));
        const winner = top.name;
        const topVal = top.value;
        
        const statusMessage = 
            `🔍 COMPARISON RESULTS\n${idInfo}Target Column: ${data.dataset.target}\n` +
            `Dataset: ${data.dataset.numFeaturesTotal} features, ${data.dataset.rows} rows\n\n` +
            `🧬 GENETIC ALGORITHM:\n` +
            `  ${gaConvergence}\n` +
            `  Accuracy: ${data.ga.accuracy}\n` +
            `  Exec Time: ${data.ga.execTimeSeconds}s\n` +
            `  Selected: ${data.ga.numFeaturesSelected} features\n` +
            `  Features: ${data.ga.selectedFeatures.join(', ')}\n\n` +
            `📊 VARIANCE THRESHOLD:\n` +
            `  Threshold: ${data.varianceThreshold.thresholdUsed}\n` +
            `  Accuracy: ${data.varianceThreshold.accuracy}\n` +
            `  Exec Time: ${data.varianceThreshold.execTimeSeconds}s\n` +
            `  Selected: ${data.varianceThreshold.numFeaturesSelected} features\n` +
            `  Features: ${data.varianceThreshold.selectedFeatures.join(', ')}\n` +
            `  Removed: ${data.varianceThreshold.removedFeatures.join(', ')}\n\n` +
            `🧪 SELECTKBEST:\n` +
            `  Score: ${data.selectKBest.scoreFuncUsed}, k=${data.selectKBest.kUsed}\n` +
            `  Accuracy: ${data.selectKBest.accuracy}\n` +
            `  Exec Time: ${data.selectKBest.execTimeSeconds}s\n` +
            `  Selected: ${data.selectKBest.numFeaturesSelected} features\n` +
            `  Features: ${data.selectKBest.selectedFeatures.join(', ')}\n` +
            `  Removed: ${data.selectKBest.removedFeatures.join(', ')}\n\n` +
            `🏆 WINNER: ${winner} (${topVal})` +
            `${feSummary}`;

        this.updateStatus(statusMessage, 'success');
        
        // Update chart
        if (this.chartManager) {
            this.chartManager.renderComparisonChart(data);
        }

        this.showToast(`Comparison complete! Winner: ${winner}`, 'success');
    }

    handleError(error) {
        console.error('Application error:', error);
        
        const errorMessage = error.message || 'An unexpected error occurred';
        this.updateStatus(`❌ Error: ${errorMessage}`, 'error');
        this.showToast(errorMessage, 'error');
    }

    // File handling
    onFileUploaded(file, csvData) {
        this.state.currentFile = file;
        this.state.csvData = csvData;
        
        const lines = csvData.split('\n').filter(x => x.trim());
        const headers = lines[0].split(',');
        
        // Show columns with their indices
        const columnInfo = headers.map((h, i) => `[${i}] ${h.trim()}`).join(', ');
        
        // Update persistent file info in upload component
        this.updateFileInfo(file, headers, lines.length - 1);
        
        // Update status
        this.updateStatus(`File "${file.name}" loaded successfully. Ready to run analysis.`, 'success');
        
        // Show column selection section
        const columnSelection = document.getElementById('columnSelection');
        if (columnSelection) {
            columnSelection.style.display = 'block';
        }
        
        // Set default values: last column as target
        const targetColumn = document.getElementById('targetColumn');
        if (targetColumn) {
            targetColumn.value = headers.length - 1;
        }
        
        const idColumn = document.getElementById('idColumn');
        if (idColumn) {
            idColumn.value = '';
        }
        
        this.showToast(`File "${file.name}" uploaded successfully!`, 'success');
    }

    updateFileInfo(file, headers, rowCount) {
        const datasetInfo = document.getElementById('datasetInfo');
        const fileDetails = document.getElementById('fileDetails');
        
        if (datasetInfo && fileDetails) {
            datasetInfo.classList.remove('hidden');
            
            // Limit column display to first 10 columns
            const maxColumns = 10;
            const showAll = headers.length <= maxColumns;
            const displayHeaders = showAll ? headers : headers.slice(0, maxColumns);
            
            const columnList = displayHeaders.map((h, i) => 
                `<span class="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded text-xs mr-1 mb-1">[${i}] ${h.trim()}</span>`
            ).join('');
            
            // Add "show more" button if there are more columns
            const showMoreButton = !showAll ? 
                `<button id="showAllColumns" class="inline-block bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded text-xs mr-1 mb-1 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">+${headers.length - maxColumns} more...</button>` : '';
            
            fileDetails.innerHTML = `
                <div class="grid grid-cols-2 gap-4 mb-3">
                    <div>
                        <span class="font-medium">File:</span> ${file.name}
                    </div>
                    <div>
                        <span class="font-medium">Size:</span> ${(file.size / 1024).toFixed(1)} KB
                    </div>
                    <div>
                        <span class="font-medium">Rows:</span> ${rowCount}
                    </div>
                    <div>
                        <span class="font-medium">Columns:</span> ${headers.length}
                    </div>
                </div>
                <div>
                    <span class="font-medium">Columns:</span>
                    <div class="mt-1">${columnList}${showMoreButton}</div>
                </div>
            `;
            
            // Add event listener for "show more" button
            if (!showAll) {
                const showAllButton = document.getElementById('showAllColumns');
                if (showAllButton) {
                    showAllButton.addEventListener('click', () => {
                        const allColumnList = headers.map((h, i) => 
                            `<span class="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded text-xs mr-1 mb-1">[${i}] ${h.trim()}</span>`
                        ).join('');
                        
                        const columnsDiv = showAllButton.parentElement;
                        columnsDiv.innerHTML = allColumnList;
                    });
                }
            }
        }
    }

    clearFile() {
        const fileInput = document.getElementById('csvFile');
        const datasetInfo = document.getElementById('datasetInfo');
        const columnSelection = document.getElementById('columnSelection');
        
        if (fileInput) fileInput.value = '';
        if (datasetInfo) datasetInfo.classList.add('hidden');
        if (columnSelection) columnSelection.style.display = 'none';
        
        this.state.csvData = '';
        this.state.currentFile = null;
        
        this.updateStatus('No file selected. Please upload a CSV file.');
        this.showToast('File cleared', 'info');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.gaApp = new GAApplication();
});

// Global functions for backward compatibility
function runGA() {
    if (window.gaApp) {
        window.gaApp.runGA();
    } else {
        console.error('GA Application not initialized');
    }
}

function runVarianceThreshold() {
    if (window.gaApp) {
        window.gaApp.runVarianceThreshold();
    } else {
        console.error('GA Application not initialized');
    }
}

function runSelectKBest() {
    if (window.gaApp) {
        window.gaApp.runSelectKBest();
    } else {
        console.error('GA Application not initialized');
    }
}

function runComparison() {
    if (window.gaApp) {
        window.gaApp.runComparison();
    } else {
        console.error('GA Application not initialized');
    }
}

function clearFile() {
    if (window.gaApp) {
        window.gaApp.clearFile();
    } else {
        console.error('GA Application not initialized');
    }
}

