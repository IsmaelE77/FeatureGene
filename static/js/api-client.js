/**
 * API Client for communicating with Flask backend
 * Handles all HTTP requests and response processing
 */

class APIClient {
    constructor(app) {
        this.app = app;
        this.baseURL = window.location.origin;
    }


    async listDatasets() {
        try {
            const response = await fetch(`${this.baseURL}/list_datasets`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            return result.datasets || [];
        } catch (error) {
            console.error('Failed to list datasets:', error);
            throw error;
        }
    }

    async getDataset(name) {
        try {
            const response = await fetch(`${this.baseURL}/get_dataset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            return result;
        } catch (error) {
            console.error('Failed to fetch dataset:', error);
            throw error;
        }
    }
    
    async makeRequest(endpoint, data) {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async runGA() {
        const parameters = this.getGAParameters();
        const requestData = {
            ...parameters,
            csvData: this.app.state.csvData
        };

        this.app.updateStatus(['Running Genetic Algorithm...', 'loading']);
        return await this.makeRequest('/run_ga', requestData);
    }

    async runVarianceThreshold() {
        const parameters = this.getVarianceThresholdParameters();
        const requestData = {
            ...parameters,
            csvData: this.app.state.csvData
        };

        this.app.updateStatus('Running Variance Threshold...', 'loading');
        return await this.makeRequest('/run_variance_threshold', requestData);
    }

    async runSelectKBest() {
        const parameters = this.getSelectKBestParameters();
        const requestData = {
            ...parameters,
            csvData: this.app.state.csvData
        };

        this.app.updateStatus('Running SelectKBest...', 'loading');
        return await this.makeRequest('/run_select_kbest', requestData);
    }

    async runComparison() {
        const gaParams = this.getGAParameters();
        const vtParams = this.getVarianceThresholdParameters();
        const kbParams = this.getSelectKBestParameters ? this.getSelectKBestParameters() : {};
        const requestData = {
            ...gaParams,
            ...vtParams,
            ...kbParams,
            csvData: this.app.state.csvData
        };

        this.app.updateStatus('Running comparison (GA vs Variance Threshold vs SelectKBest)...', 'loading');
        return await this.makeRequest('/run_comparison', requestData);
    }

    getGAParameters() {
        const popSize = document.getElementById('popSize')?.value || '30';
        const crossRate = document.getElementById('crossRate')?.value || '0.7';
        const mutRate = document.getElementById('mutRate')?.value || '0.1';
        const maxGen = document.getElementById('maxGen')?.value || '20';
        const convergenceThreshold = document.getElementById('convergenceThreshold')?.value || '';
        const idColumn = document.getElementById('idColumn')?.value || '';
        const targetColumn = document.getElementById('targetColumn')?.value;

        const params = {
            popSize: parseInt(popSize),
            crossRate: parseFloat(crossRate),
            mutRate: parseFloat(mutRate),
            maxGen: parseInt(maxGen),
            idColumn: idColumn === '' ? null : parseInt(idColumn),
            targetColumn: parseInt(targetColumn)
        };

        // Only include convergenceThreshold if it has a value
        if (convergenceThreshold !== '') {
            params.convergenceThreshold = parseFloat(convergenceThreshold);
        }

        return params;
    }

    getVarianceThresholdParameters() {
        const vtThreshold = document.getElementById('vtThreshold')?.value || '0.0';
        const idColumn = document.getElementById('idColumn')?.value || '';
        const targetColumn = document.getElementById('targetColumn')?.value;

        return {
            threshold: parseFloat(vtThreshold),
            idColumn: idColumn === '' ? null : parseInt(idColumn),
            targetColumn: parseInt(targetColumn)
        };
    }

    getSelectKBestParameters() {
        const kVal = document.getElementById('kbestK')?.value || '';
        const scoreFunc = document.getElementById('kbestScoreFunc')?.value || 'f_classif';
        const idColumn = document.getElementById('idColumn')?.value || '';
        const targetColumn = document.getElementById('targetColumn')?.value;

        return {
            k: kVal === '' ? null : parseInt(kVal),
            scoreFunc: scoreFunc,
            idColumn: idColumn === '' ? null : parseInt(idColumn),
            targetColumn: parseInt(targetColumn)
        };
    }

    // Utility method to validate parameters before sending
    validateParameters() {
        const errors = [];

        // Check required fields
        const targetColumn = document.getElementById('targetColumn')?.value;
        if (!targetColumn && targetColumn !== '0') {
            errors.push('Target column is required');
        }

        // Validate numeric parameters
        const popSize = parseInt(document.getElementById('popSize')?.value || '30');
        if (popSize < 2 || popSize > 1000) {
            errors.push('Population size must be between 2 and 1000');
        }

        const crossRate = parseFloat(document.getElementById('crossRate')?.value || '0.7');
        if (crossRate < 0 || crossRate > 1) {
            errors.push('Crossover rate must be between 0 and 1');
        }

        const mutRate = parseFloat(document.getElementById('mutRate')?.value || '0.1');
        if (mutRate < 0 || mutRate > 1) {
            errors.push('Mutation rate must be between 0 and 1');
        }

        const maxGen = parseInt(document.getElementById('maxGen')?.value || '20');
        if (maxGen < 1 || maxGen > 1000) {
            errors.push('Max generations must be between 1 and 1000');
        }

        const vtThreshold = parseFloat(document.getElementById('vtThreshold')?.value || '0.0');
        if (vtThreshold < 0) {
            errors.push('Variance threshold must be non-negative');
        }

        return errors;
    }

    // Method to get current request status
    getRequestStatus() {
        return {
            hasData: !!this.app.state.csvData,
            isProcessing: this.app.state.isProcessing,
            parameters: this.getGAParameters()
        };
    }

    // Method to cancel current request (if needed)
    cancelRequest() {
        // Note: This is a placeholder. Actual implementation would require
        // AbortController for fetch requests
        console.log('Request cancellation not implemented');
    }

    // Method to retry last failed request
    async retryLastRequest() {
        if (!this.app.state.lastRequest) {
            throw new Error('No previous request to retry');
        }

        const { endpoint, data } = this.app.state.lastRequest;
        return await this.makeRequest(endpoint, data);
    }

    // Enhanced error handling
    handleAPIError(error, context = '') {
        let userMessage = 'An error occurred';
        
        if (error.message.includes('Failed to fetch')) {
            userMessage = 'Unable to connect to server. Please check your connection.';
        } else if (error.message.includes('HTTP error')) {
            userMessage = 'Server error occurred. Please try again.';
        } else if (error.message.includes('Target column')) {
            userMessage = error.message;
        } else if (error.message.includes('CSV')) {
            userMessage = error.message;
        } else {
            userMessage = error.message || 'An unexpected error occurred';
        }

        if (context) {
            userMessage = `${context}: ${userMessage}`;
        }

        return userMessage;
    }
}
