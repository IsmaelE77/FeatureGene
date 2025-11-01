/**
 * Chart Manager for handling Chart.js visualizations
 * Manages different chart types and themes
 */

class ChartManager {
    constructor(app) {
        this.app = app;
        this.currentChart = null; // can be Chart instance or Array<Chart>
        // Look for chart container first, then canvas
        this.chartContainer = document.querySelector('.chart-container') || document.getElementById('chart');
        this.setupChartContainer();
    }

    setupChartContainer() {
        if (!this.chartContainer) {
            console.warn('Chart container not found');
            return;
        }

        // Set initial dimensions
        this.chartContainer.style.width = '100%';
        this.chartContainer.style.height = '400px';
        this.chartContainer.style.position = 'relative';
        
        // Check if canvas already exists
        const existingCanvas = this.chartContainer.querySelector('canvas');
        
        if (!existingCanvas) {
            // Create canvas if it doesn't exist
            const canvas = document.createElement('canvas');
            canvas.id = 'chart';
            this.chartContainer.appendChild(canvas);
        }
    }

    destroyCurrentChart() {
        if (!this.currentChart) return;
        // Support single or multiple charts
        if (Array.isArray(this.currentChart)) {
            this.currentChart.forEach(ch => {
                try { ch.destroy(); } catch (e) { /* noop */ }
            });
        } else {
            try { this.currentChart.destroy(); } catch (e) { /* noop */ }
        }
        this.currentChart = null;
    }

    showLoadingState() {
        if (this.chartContainer) {
            this.chartContainer.innerHTML = `
                <div class="chart-loading">
                    <div class="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                    <span class="ml-2 text-gray-500">Loading chart...</span>
                </div>
            `;
        }
    }

    renderGAChart(data) {
        this.destroyCurrentChart();

        if (!this.chartContainer) {
            console.error('Chart container not found');
            return;
        }

        // Ensure container height fits one chart
        this.chartContainer.style.height = '400px';
        // Reset container to a single canvas layout
        this.chartContainer.innerHTML = '<canvas id="chart"></canvas>';
        // Get canvas
        const canvas = this.chartContainer.querySelector('canvas');
        if (!canvas) {
            console.error('Canvas not found in chart container');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.history.map((_, i) => `Gen ${i + 1}`),
                datasets: [{
                    label: 'Best Fitness',
                    data: data.history,
                    borderColor: this.getThemeColor('primary'),
                    backgroundColor: this.getThemeColor('primary', 0.1),
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.getThemeColor('primary'),
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Genetic Algorithm Fitness Progress',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        color: this.getThemeColor('text')
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: this.getThemeColor('text'),
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: this.getThemeColor('background'),
                        titleColor: this.getThemeColor('text'),
                        bodyColor: this.getThemeColor('text'),
                        borderColor: this.getThemeColor('border'),
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Generation',
                            color: this.getThemeColor('text')
                        },
                        grid: {
                            color: this.getThemeColor('grid'),
                            drawBorder: false
                        },
                        ticks: {
                            color: this.getThemeColor('text')
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Fitness Score',
                            color: this.getThemeColor('text')
                        },
                        grid: {
                            color: this.getThemeColor('grid'),
                            drawBorder: false
                        },
                        ticks: {
                            color: this.getThemeColor('text'),
                            callback: function(value) {
                                return value.toFixed(3);
                            }
                        },
                        beginAtZero: true,
                        max: 1
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    renderVarianceThresholdChart(data) {
        this.destroyCurrentChart();

        if (!this.chartContainer) return;

        // Ensure container height fits one chart
        this.chartContainer.style.height = '400px';
        // Reset container to a single canvas layout
        this.chartContainer.innerHTML = '<canvas id="chart"></canvas>';
        // Get canvas
        const canvas = this.chartContainer.querySelector('canvas');
        if (!canvas) {
            console.error('Canvas not found in chart container');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        this.currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Selected Features', 'Removed Features'],
                datasets: [{
                    label: 'Feature Count',
                    data: [data.numFeaturesSelected, data.removedFeatures.length],
                    backgroundColor: [
                        this.getThemeColor('success'),
                        this.getThemeColor('error')
                    ],
                    borderColor: [
                        this.getThemeColor('success', 0.8),
                        this.getThemeColor('error', 0.8)
                    ],
                    borderWidth: 1,
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Variance Threshold Feature Selection',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        color: this.getThemeColor('text')
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: this.getThemeColor('background'),
                        titleColor: this.getThemeColor('text'),
                        bodyColor: this.getThemeColor('text'),
                        borderColor: this.getThemeColor('border'),
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                return `${label}: ${value} features`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: this.getThemeColor('text'),
                            font: {
                                weight: 'bold'
                            }
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Number of Features',
                            color: this.getThemeColor('text')
                        },
                        grid: {
                            color: this.getThemeColor('grid'),
                            drawBorder: false
                        },
                        ticks: {
                            color: this.getThemeColor('text'),
                            stepSize: 1
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    renderComparisonChart(data) {
        this.destroyCurrentChart();

        if (!this.chartContainer) return;

        // Increase container height to fit two stacked charts
        this.chartContainer.style.height = '650px';
        // Build two separate canvases: GA history and final accuracy comparison
        this.chartContainer.innerHTML = `
            <div class="space-y-6">
                <div style="position: relative; height: 300px;">
                    <canvas id="ga-history-chart"></canvas>
                </div>
                <div style="position: relative; height: 300px;">
                    <canvas id="final-accuracy-chart"></canvas>
                </div>
            </div>
        `;

        // Chart 1: GA history (line)
        const ctx1 = this.chartContainer.querySelector('#ga-history-chart').getContext('2d');
        const gaHistoryChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: data.ga.history.map((_, i) => `Gen ${i + 1}`),
                datasets: [{
                    label: 'GA Fitness Progress',
                    data: data.ga.history,
                    borderColor: this.getThemeColor('primary'),
                    backgroundColor: this.getThemeColor('primary', 0.1),
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.getThemeColor('primary'),
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'GA Fitness Progress',
                        color: this.getThemeColor('text')
                    },
                    legend: {
                        labels: { color: this.getThemeColor('text') }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Generation', color: this.getThemeColor('text') },
                         ticks: { color: this.getThemeColor('text') },
                         grid: { color: this.getThemeColor('grid'), drawBorder: false } },
                    y: { title: { display: true, text: 'Fitness', color: this.getThemeColor('text') },
                         ticks: { color: this.getThemeColor('text'), callback: v => v.toFixed(3) },
                         grid: { color: this.getThemeColor('grid'), drawBorder: false },
                         beginAtZero: true, max: 1 }
                }
            }
        });

        // Chart 2: Final accuracy comparison (bar)
        const ctx2 = this.chartContainer.querySelector('#final-accuracy-chart').getContext('2d');
        const finalCompareChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['GA (final)', 'Variance Threshold'],
                datasets: [{
                    label: 'Accuracy',
                    data: [data.ga.accuracy, data.varianceThreshold.accuracy],
                    backgroundColor: [this.getThemeColor('primary', 0.6), this.getThemeColor('secondary', 0.6)],
                    borderColor: [this.getThemeColor('primary'), this.getThemeColor('secondary')],
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Final Accuracy: GA vs Variance Threshold',
                        color: this.getThemeColor('text')
                    },
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Accuracy: ${ctx.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: this.getThemeColor('text') } },
                    y: { beginAtZero: true, max: 1,
                         ticks: { color: this.getThemeColor('text'), callback: v => v.toFixed(2) },
                         grid: { color: this.getThemeColor('grid'), drawBorder: false },
                         title: { display: true, text: 'Accuracy', color: this.getThemeColor('text') } }
                }
            }
        });

        // Save both charts
        this.currentChart = [gaHistoryChart, finalCompareChart];
    }

    updateTheme(theme) {
        if (!this.currentChart) return;

        const applyTheme = (chart) => {
            if (!chart || !chart.options) return;
            const isDark = document.documentElement.classList.contains('dark');
            const text = isDark ? '#e5e7eb' : '#374151';
            const grid = isDark ? '#374151' : '#e5e7eb';
            if (chart.options.plugins && chart.options.plugins.title) {
                chart.options.plugins.title.color = text;
            }
            if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
                chart.options.plugins.legend.labels.color = text;
            }
            if (chart.options.scales) {
                const { x, y } = chart.options.scales;
                if (x) {
                    if (x.title) x.title.color = text;
                    if (x.ticks) x.ticks.color = text;
                    if (x.grid) x.grid.color = grid;
                }
                if (y) {
                    if (y.title) y.title.color = text;
                    if (y.ticks) y.ticks.color = text;
                    if (y.grid) y.grid.color = grid;
                }
            }
            chart.update();
        };

        if (Array.isArray(this.currentChart)) {
            this.currentChart.forEach(applyTheme);
        } else {
            applyTheme(this.currentChart);
        }
    }

    getThemeColor(colorType, alpha = 1) {
        const isDark = document.documentElement.classList.contains('dark');
        
        const colors = {
            light: {
                primary: `rgba(59, 130, 246, ${alpha})`,
                secondary: `rgba(239, 68, 68, ${alpha})`,
                success: `rgba(16, 185, 129, ${alpha})`,
                error: `rgba(239, 68, 68, ${alpha})`,
                warning: `rgba(245, 158, 11, ${alpha})`,
                text: '#374151',
                background: '#ffffff',
                border: '#e5e7eb',
                grid: '#e5e7eb'
            },
            dark: {
                primary: `rgba(96, 165, 250, ${alpha})`,
                secondary: `rgba(248, 113, 113, ${alpha})`,
                success: `rgba(52, 211, 153, ${alpha})`,
                error: `rgba(248, 113, 113, ${alpha})`,
                warning: `rgba(251, 191, 36, ${alpha})`,
                text: '#e5e7eb',
                background: '#1f2937',
                border: '#374151',
                grid: '#374151'
            }
        };

        return colors[isDark ? 'dark' : 'light'][colorType] || colors.light[colorType];
    }

    exportChart(format = 'png') {
        if (!this.currentChart) {
            this.app.showToast('No chart to export', 'warning');
            return;
        }

        // If multiple charts, export the first one (GA history)
        const chartToExport = Array.isArray(this.currentChart) ? this.currentChart[0] : this.currentChart;
        const link = document.createElement('a');
        link.download = `ga-chart-${new Date().toISOString().split('T')[0]}.${format}`;
        link.href = chartToExport.toBase64Image();
        link.click();
        
        this.app.showToast(`Chart exported as ${format.toUpperCase()}`, 'success');
    }

    resizeChart() {
        if (!this.currentChart) return;
        if (Array.isArray(this.currentChart)) {
            this.currentChart.forEach(ch => ch && ch.resize());
        } else {
            this.currentChart.resize();
        }
    }
}
