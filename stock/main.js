let mainChart = null;
let rsiChart = null;

const predictForm = document.getElementById('predict-form');
const loadingOverlay = document.getElementById('loading-overlay');
const lastUpdated = document.getElementById('last-updated');

const kpiCurrentPrice = document.getElementById('kpi-current-price');
const kpiPredictedPrice = document.getElementById('kpi-predicted-price');
const kpiPriceChange = document.getElementById('kpi-price-change');
const kpiSignal = document.getElementById('kpi-signal');
const kpiRmse = document.getElementById('kpi-rmse');
const kpiMape = document.getElementById('kpi-mape');
const kpiPredictedBg = document.getElementById('kpi-predicted-bg');

const lookbackInput = document.getElementById('lookback');
const lookbackVal = document.getElementById('lookback-val');
const epochsInput = document.getElementById('epochs');
const epochsVal = document.getElementById('epochs-val');
const toggleSma = document.getElementById('toggle-sma');

lookbackInput.addEventListener('input', (e) => {
    lookbackVal.textContent = e.target.value;
});

epochsInput.addEventListener('input', (e) => {
    epochsVal.textContent = e.target.value;
});

toggleSma.addEventListener('change', (e) => {
    if (!mainChart) return;
    if (e.target.checked) {
        mainChart.showSeries('SMA 20');
        mainChart.showSeries('SMA 50');
    } else {
        mainChart.hideSeries('SMA 20');
        mainChart.hideSeries('SMA 50');
    }
});

predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    loadingOverlay.classList.remove('hidden');
    loadingOverlay.classList.add('flex');

    try {
        const response = await fetch('http://127.0.0.1:8000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: document.getElementById('ticker').value,
                lookback: parseInt(lookbackInput.value),
                epochs: parseInt(epochsInput.value),
                years: 5
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'API Request Failed');
        }

        const data = await response.json();
        updateDashboard(data);

    } catch (error) {
        console.error(error);
        alert(`Error executing model: ${error.message}`);
    } finally {
        loadingOverlay.classList.add('hidden');
        loadingOverlay.classList.remove('flex');
    }
});

function updateDashboard(data) {
    const { metrics, charts } = data;

    kpiCurrentPrice.textContent = `$${metrics.current_price.toFixed(2)}`;
    kpiPredictedPrice.textContent = `$${metrics.predicted_next_price.toFixed(2)}`;
    
    const change = metrics.predicted_next_price - metrics.current_price;
    const pctChange = (change / metrics.current_price) * 100;
    
    if (change > 0) {
        kpiPriceChange.textContent = `+${pctChange.toFixed(2)}% ▲`;
        kpiPriceChange.className = 'text-sm font-semibold text-brand-emerald';
        kpiPredictedBg.className = 'absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl -mr-10 -mt-10 transition-colors opacity-50 bg-brand-emerald/30';
    } else {
        kpiPriceChange.textContent = `${pctChange.toFixed(2)}% ▼`;
        kpiPriceChange.className = 'text-sm font-semibold text-brand-crimson';
        kpiPredictedBg.className = 'absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl -mr-10 -mt-10 transition-colors opacity-50 bg-brand-crimson/30';
    }

    kpiRmse.textContent = `$${metrics.rmse.toFixed(2)}`;
    kpiMape.textContent = `${(metrics.mape * 100).toFixed(2)}%`;

    kpiSignal.textContent = metrics.direction_signal;
    if (metrics.direction_signal === 'BUY') {
        kpiSignal.className = 'mt-4 px-6 py-2 rounded-full font-black tracking-widest text-xl border uppercase bg-brand-emerald/20 text-brand-emerald border-brand-emerald';
    } else if (metrics.direction_signal === 'SELL') {
        kpiSignal.className = 'mt-4 px-6 py-2 rounded-full font-black tracking-widest text-xl border uppercase bg-brand-crimson/20 text-brand-crimson border-brand-crimson';
    } else {
        kpiSignal.className = 'mt-4 px-6 py-2 rounded-full font-black tracking-widest text-xl border uppercase bg-gray-800 text-gray-500 border-gray-700';
    }

    lastUpdated.textContent = `Last executed: ${new Date().toLocaleTimeString()}`;

    renderCharts(charts);
}

function renderCharts(data) {
    const dates = data.dates;

    const actualSeries = data.actual_prices.map((val, i) => ({ x: dates[i], y: val }));
    const predictedSeries = data.predicted_prices.map((val, i) => ({ x: dates[i], y: val }));
    const sma20Series = data.sma20.map((val, i) => ({ x: dates[i], y: val }));
    const sma50Series = data.sma50.map((val, i) => ({ x: dates[i], y: val }));
    const rsiSeries = data.rsi.map((val, i) => ({ x: dates[i], y: val }));

    const mainOptions = {
        series: [
            { name: 'Actual Price', type: 'area', data: actualSeries },
            { name: 'Predicted Forecast', type: 'line', data: predictedSeries },
            { name: 'SMA 20', type: 'line', data: sma20Series },
            { name: 'SMA 50', type: 'line', data: sma50Series }
        ],
        chart: {
            height: 400,
            type: 'line',
            background: 'transparent',
            toolbar: { show: true },
            animations: { enabled: false },
            id: 'price-chart',
            group: 'sync-charts'
        },
        colors: ['#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6'],
        fill: {
            type: ['gradient', 'solid', 'solid', 'solid'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.0,
                stops: [0, 100]
            }
        },
        stroke: {
            width: [2, 3, 1, 1],
            curve: 'straight',
            dashArray: [0, 5, 0, 0]
        },
        xaxis: {
            type: 'datetime',
            labels: { style: { colors: '#94a3b8' } },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            labels: {
                formatter: (value) => `$${value?.toFixed(2)}`,
                style: { colors: '#94a3b8' }
            }
        },
        theme: { mode: 'dark' },
        grid: {
            borderColor: '#334155',
            strokeDashArray: 4,
        },
        dataLabels: { enabled: false },
        legend: {
            position: 'top',
            horizontalAlign: 'left'
        }
    };

    const rsiOptions = {
        series: [{ name: 'RSI (14)', data: rsiSeries }],
        chart: {
            height: 200,
            type: 'line',
            background: 'transparent',
            toolbar: { show: false },
            animations: { enabled: false },
            id: 'rsi-chart',
            group: 'sync-charts'
        },
        colors: ['#06b6d4'],
        stroke: { width: 2 },
        xaxis: {
            type: 'datetime',
            labels: { show: false },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            min: 0,
            max: 100,
            tickAmount: 4,
            labels: { style: { colors: '#94a3b8' } }
        },
        annotations: {
            yaxis: [
                { y: 70, borderColor: '#ef4444', label: { text: 'Overbought', style: { color: '#fff', background: '#ef4444' } } },
                { y: 30, borderColor: '#10b981', label: { text: 'Oversold', style: { color: '#fff', background: '#10b981' } } }
            ]
        },
        theme: { mode: 'dark' },
        grid: {
            borderColor: '#334155',
            strokeDashArray: 4,
        },
        dataLabels: { enabled: false }
    };

    if (mainChart) {
        mainChart.updateOptions(mainOptions);
    } else {
        mainChart = new ApexCharts(document.querySelector("#main-chart"), mainOptions);
        mainChart.render();
    }

    if (!toggleSma.checked && mainChart) {
        mainChart.hideSeries('SMA 20');
        mainChart.hideSeries('SMA 50');
    }

    if (rsiChart) {
        rsiChart.updateOptions(rsiOptions);
    } else {
        rsiChart = new ApexCharts(document.querySelector("#rsi-chart"), rsiOptions);
        rsiChart.render();
    }
}
