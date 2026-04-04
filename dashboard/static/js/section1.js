let currentFilters = {};
let allData = {};

const PLOT_CONFIG = { responsive: true, displayModeBar: false };

/** Map: wheel zoom + double-click reset; tight legend text */
const MAP_PLOT_CONFIG = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
    doubleClick: 'reset',
};

function mergeTightMargins(layout, overrides = {}) {
    const base = { l: 32, r: 10, t: 36, b: 32, pad: 0 };
    const m = layout && layout.margin ? { ...base, ...layout.margin, ...overrides } : { ...base, ...overrides };
    return { ...(layout || {}), margin: m };
}

function relayoutPlotlyIn(container) {
    const gd = container?.querySelector('.js-plotly-plot');
    if (!gd) return;
    Plotly.relayout(gd, { margin: { l: 32, r: 10, t: 34, b: 30, pad: 0 } });
}

function resizePlotlyDiv(containerId) {
    const el = document.getElementById(containerId);
    const gd = el?.querySelector?.('.js-plotly-plot');
    if (gd) Plotly.Plots.resize(gd);
}

function resizeAllSection1Plots() {
    resizePlotlyDiv('plot1-container');
    resizePlotlyDiv('plot2-container');
    resizePlotlyDiv('map-container');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Get filter column names
    const filterCols = {
        "order_birdlife": "Order",
        "family_birdlife": "Family",
        "trophic_niche": "Trophic Niche",
        "habitat_density": "Habitat Density",
    };
    
    // Build filter UI
    const filtersContainer = document.getElementById('filters-container');
    for (const [col, label] of Object.entries(filterCols)) {
        const filterDiv = document.createElement('div');
        filterDiv.className = 'filter-row';
        filterDiv.innerHTML = `
            <label>${label}:</label>
            <select data-filter="${col}" class="filter-select">
                <option value="">All</option>
            </select>
        `;
        filtersContainer.appendChild(filterDiv);
        
        // Load filter values
        try {
            const response = await fetch(`/api/filter-values/${col}`);
            const data = await response.json();
            const select = filterDiv.querySelector('.filter-select');
            data.values.forEach(val => {
                const option = document.createElement('option');
                option.value = val;
                option.textContent = val;
                select.appendChild(option);
            });
            
            // Add change listener
            select.addEventListener('change', () => {
                const col = select.dataset.filter;
                if (select.value) {
                    currentFilters[col] = select.value;
                } else {
                    delete currentFilters[col];
                }
                updateAnalysis();
                updateFilterBadges();
            });
        } catch (error) {
            console.error(`Error loading filter values for ${col}:`, error);
        }
    }
    
    // Feature select
    document.getElementById('feature-select').addEventListener('change', (e) => {
        const feature = e.target.value;
        const logScaleGroup = document.getElementById('log-scale-group');
        const aggregateGroup = document.getElementById('aggregate-line-group');
        
        // Show log scale and aggregate toggle only for numerical features
        if (NUMERICAL_TRAITS.includes(feature)) {
            logScaleGroup.style.display = 'block';
            aggregateGroup.style.display = 'block';
        } else {
            logScaleGroup.style.display = 'none';
            aggregateGroup.style.display = 'none';
            document.getElementById('use-log').checked = false;
            document.getElementById('show-aggregate-line').checked = false;
        }
        
        // Reset subfeature
        document.getElementById('subfeature-select').value = '';
        updateAnalysis();
    });
    
    // Log scale toggle
    document.getElementById('use-log').addEventListener('change', () => {
        updateAnalysis();
    });
    
    // Aggregate line toggle
    document.getElementById('show-aggregate-line').addEventListener('change', () => {
        updateAnalysis();
    });
    
    // Graph type select
    document.getElementById('graph-type-select').addEventListener('change', () => {
        updateAnalysis();
    });

    // Sub-feature select
    document.getElementById('subfeature-select').addEventListener('change', () => {
        updateAnalysis();
    });

    // Lat/lon slider listeners
    ['lat-min', 'lat-max', 'lon-min', 'lon-max'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => {
            document.getElementById(id + '-val').textContent = document.getElementById(id).value;
        });
        document.getElementById(id).addEventListener('change', () => {
            updateAnalysis();
            updateMap();
        });
    });

    // Map color selector
    document.getElementById('map-color-col').addEventListener('change', () => updateMap());
    updateMap();

    let resizeT;
    window.addEventListener('resize', () => {
        clearTimeout(resizeT);
        resizeT = setTimeout(() => {
            resizeAllSection1Plots();
            relayoutSection1MapToContainer();
        }, 100);
    });

    const mapHost = document.getElementById('map-container');
    if (mapHost && !mapHost.dataset.resizeObs) {
        mapHost.dataset.resizeObs = '1';
        new ResizeObserver(() => relayoutSection1MapToContainer()).observe(mapHost);
    }
});

function getMapContainerPixelSize() {
    const el = document.getElementById('map-container');
    if (!el) return { w: 400, h: 280 };
    const r = el.getBoundingClientRect();
    return {
        w: Math.max(160, Math.floor(r.width)),
        h: Math.max(140, Math.floor(r.height)),
    };
}

function relayoutSection1MapToContainer() {
    const el = document.getElementById('map-container');
    const gd = el?.querySelector('.js-plotly-plot');
    if (!gd || !gd.layout) return;
    const { w, h } = getMapContainerPixelSize();
    if (w < 20 || h < 20) return;
    Plotly.relayout(gd, { width: w, height: h }).catch(() => {});
}

async function updateMap() {
    const colorCol = document.getElementById('map-color-col').value;
    const latMin = parseInt(document.getElementById('lat-min').value);
    const latMax = parseInt(document.getElementById('lat-max').value);
    const lonMin = parseInt(document.getElementById('lon-min').value);
    const lonMax = parseInt(document.getElementById('lon-max').value);
    const filters = { ...currentFilters };
    if (latMin > -72 || latMax < 79) filters.lat_range = [latMin, latMax];
    if (lonMin > -179 || lonMax < 180) filters.lon_range = [lonMin, lonMax];
    try {
        const res = await fetch('/api/map-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ color_col: colorCol, filters }),
        });
        const data = await res.json();
        if (!res.ok) return;
        const PALETTE = ['#3B82F6','#EF4444','#10B981','#F59E0B','#8B5CF6','#06B6D4','#F97316','#EC4899','#84CC16','#6366F1',
                         '#14B8A6','#A855F7','#F43F5E','#22C55E','#FACC15','#64748B','#0EA5E9','#D946EF','#FB923C','#2DD4BF'];
        const groups = {};
        data.points.forEach(p => {
            const g = p.color_val || 'Unknown';
            if (!groups[g]) groups[g] = { lat: [], lon: [], text: [] };
            groups[g].lat.push(p.lat);
            groups[g].lon.push(p.lon);
            groups[g].text.push(p.species);
        });
        const traces = Object.entries(groups).map(([name, d], i) => ({
            type: 'scattergeo', mode: 'markers', name,
            lat: d.lat, lon: d.lon, text: d.text,
            marker: { size: 4, color: PALETTE[i % PALETTE.length], opacity: 0.7 },
            hovertemplate: '%{text}<extra>' + name + '</extra>',
        }));
        const mapEl = document.getElementById('map-container');
        const { w: mapW, h: mapH } = getMapContainerPixelSize();
        const layout = {
            margin: { l: 0, r: 0, t: 0, b: 0, pad: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            width: mapW,
            height: mapH,
            autosize: false,
            showlegend: traces.length > 0,
            legend: {
                orientation: 'h',
                xref: 'paper',
                yref: 'paper',
                x: 0.5,
                y: 0.012,
                xanchor: 'center',
                yanchor: 'bottom',
                font: { size: 7, family: 'Segoe UI, sans-serif' },
                bgcolor: 'rgba(255,255,255,0.96)',
                bordercolor: 'rgba(180,180,180,0.4)',
                borderwidth: 1,
                itemsizing: 'constant',
                itemwidth: 14,
                tracegroupgap: 1,
                traceorder: 'normal',
            },
            geo: {
                showland: true,
                landcolor: '#f0f0f0',
                coastlinecolor: '#ccc',
                projection: { type: 'natural earth' },
                bgcolor: 'rgba(0,0,0,0)',
                domain: { x: [0, 1], y: [0, 1] },
                lataxis: { range: [-90, 90], showgrid: true, gridcolor: 'rgba(0,0,0,0.06)' },
                lonaxis: { range: [-180, 180], showgrid: true, gridcolor: 'rgba(0,0,0,0.06)' },
            },
        };
        const drawMap = () => {
            const { w, h } = getMapContainerPixelSize();
            layout.width = w;
            layout.height = h;
            Plotly.react(mapEl, traces, layout, MAP_PLOT_CONFIG);
            requestAnimationFrame(() => relayoutSection1MapToContainer());
        };
        requestAnimationFrame(() => requestAnimationFrame(drawMap));
    } catch (e) {
        console.error('Map error:', e);
    }
}

function updateFilterBadges() {
    const badgesDiv = document.getElementById('filters-container');
    const existing = badgesDiv.querySelectorAll('.badge');
    existing.forEach(b => b.remove());
    
    for (const [col, value] of Object.entries(currentFilters)) {
        const badge = document.createElement('div');
        badge.className = 'badge';
        badge.innerHTML = `${value} <span class="remove">×</span>`;
        badge.onclick = () => {
            delete currentFilters[col];
            // Reset filter select
            const select = document.querySelector(`[data-filter="${col}"]`);
            select.value = '';
            updateAnalysis();
            updateFilterBadges();
        };
        badgesDiv.appendChild(badge);
    }
}

function clearAllFilters() {
    currentFilters = {};
    // Reset all filter selects
    document.querySelectorAll('.filter-select').forEach(select => {
        select.value = '';
    });
    // Reset lat/lon sliders
    document.getElementById('lat-min').value = -72;
    document.getElementById('lat-max').value = 79;
    document.getElementById('lon-min').value = -179;
    document.getElementById('lon-max').value = 180;
    document.getElementById('lat-min-val').textContent = '-72';
    document.getElementById('lat-max-val').textContent = '79';
    document.getElementById('lon-min-val').textContent = '-179';
    document.getElementById('lon-max-val').textContent = '180';
    updateFilterBadges();
    updateAnalysis();
    updateMap();
}

async function updateAnalysis() {
    const feature = document.getElementById('feature-select').value;
    
    if (!feature) {
        document.getElementById('empty-state').style.display = 'flex';
        document.getElementById('plot1-container').style.display = 'none';
        document.getElementById('plot2-container').style.display = 'none';
        document.getElementById('stats-section').style.display = 'none';
        document.getElementById('left-species-counter').style.display = 'none';
        return;
    }
    
    const rawSubfeature = document.getElementById('subfeature-select').value || null;
    const subfeature = rawSubfeature && rawSubfeature !== feature ? rawSubfeature : null;
    const useLog = document.getElementById('use-log').checked;
    const graphType = document.getElementById('graph-type-select').value;

    // Lat/lon range filters
    const latMin = parseInt(document.getElementById('lat-min').value);
    const latMax = parseInt(document.getElementById('lat-max').value);
    const lonMin = parseInt(document.getElementById('lon-min').value);
    const lonMax = parseInt(document.getElementById('lon-max').value);
    const filtersWithGeo = { ...currentFilters };
    if (latMin > -72 || latMax < 79) filtersWithGeo.lat_range = [latMin, latMax];
    if (lonMin > -179 || lonMax < 180) filtersWithGeo.lon_range = [lonMin, lonMax];
    
    document.getElementById('loading').classList.add('active');
    document.getElementById('empty-state').style.display = 'none';
    
    try {
        const response = await fetch('/api/analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                feature: feature,
                subfeature: subfeature,
                use_log: useLog,
                graph_type: graphType,
                filters: filtersWithGeo
            })
        });
        
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        document.getElementById('left-filtered-count').textContent = data.filtered_count;
        document.getElementById('left-total-count').textContent = data.total_count;
        document.getElementById('left-species-counter').style.display = 'block';

        // Gracefully handle empty filtered result (no species).
        if (data.no_data) {
            document.getElementById('plot1-container').style.display = 'none';
            document.getElementById('plot1-container').innerHTML = '';
            document.getElementById('plot2-container').style.display = 'none';
            document.getElementById('plot2-container').innerHTML = '';
            document.getElementById('plot2-stats').style.display = 'none';
            document.getElementById('plot2-stats').innerHTML = '';
            document.getElementById('stats-section').style.display = 'none';
            document.getElementById('top-categories').style.display = 'none';
            document.getElementById('related-features').style.display = 'none';
            document.getElementById('empty-state').textContent = data.message || 'There is no species for selected filters.';
            document.getElementById('empty-state').style.display = 'flex';
            return;
        }
        
        // Update Plot 1
        const plot1Container = document.getElementById('plot1-container');
        plot1Container.innerHTML = data.plot1;
        plot1Container.style.display = 'block';
        executePlotlyScripts(plot1Container, () => {
            relayoutPlotlyIn(plot1Container);
            resizePlotlyDiv('plot1-container');
        });
        
        // Update Plot 2
        const plot2Container = document.getElementById('plot2-container');
        const plot2Stats = document.getElementById('plot2-stats');
        if (data.plot2_figure) {
            plot2Container.innerHTML = '';
            plot2Container.style.display = 'block';
            const p2layout = mergeTightMargins(data.plot2_figure.layout);
            Plotly.react(plot2Container, data.plot2_figure.data, p2layout, PLOT_CONFIG);
            requestAnimationFrame(() => resizePlotlyDiv('plot2-container'));

            if (data.plot2_stats) {
                const statEntries = Object.entries(data.plot2_stats);
                const hasNumericStats = statEntries.some(([, stats]) =>
                    ['mean', 'median', 'min', 'max', 'q1', 'q3'].every(key => typeof stats[key] === 'number')
                );

                const statRows = statEntries.map(([group, stats]) => {
                    if (hasNumericStats) {
                        return `
                            <tr>
                                <td>${group}</td>
                                <td>${stats.mean.toFixed(2)}</td>
                                <td>${stats.median.toFixed(2)}</td>
                                <td>${stats.min.toFixed(2)}</td>
                                <td>${stats.max.toFixed(2)}</td>
                                <td>${stats.q1.toFixed(2)}</td>
                                <td>${stats.q3.toFixed(2)}</td>
                            </tr>`;
                    }
                    return `
                        <tr>
                            <td>${group}</td>
                            <td>${stats.count ?? 0}</td>
                        </tr>`;
                }).join('');

                const statHeader = hasNumericStats
                    ? `
                        <tr>
                            <th>Group</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Q1</th>
                            <th>Q3</th>
                        </tr>
                    `
                    : `
                        <tr>
                            <th>Group</th>
                            <th>Count</th>
                        </tr>
                    `;

                plot2Stats.innerHTML = `
                    <h3 style="font-size:10px; margin-bottom:4px; color:#333; font-weight:700;">Subfeature stats</h3>
                    <table class="category-table">
                        <thead>
                            ${statHeader}
                        </thead>
                        <tbody>${statRows}</tbody>
                    </table>
                `;
                plot2Stats.style.display = 'block';
            } else {
                plot2Stats.style.display = 'none';
                plot2Stats.innerHTML = '';
            }
        } else {
            plot2Container.style.display = 'none';
            plot2Stats.style.display = 'none';
            plot2Stats.innerHTML = '';
        }
        
        
        // Update Stats
        const statsDisplay = document.getElementById('stats-display');
        statsDisplay.innerHTML = '';
        for (const [key, value] of Object.entries(data.stats)) {
            const row = document.createElement('div');
            row.className = 'stat-row';
            row.innerHTML = `
                <span class="stat-label">${key}</span>
                <span class="stat-value">${value}</span>
            `;
            statsDisplay.appendChild(row);
        }
        
        // Update top categories if available
        if (data.top_categories) {
            const topCatsDiv = document.getElementById('top-categories');
            topCatsDiv.innerHTML = `
                <h3 style="font-size:10px; margin-bottom:4px; color:#555; font-weight:600;">Top categories (sub-feature)</h3>
                <table class="category-table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(data.top_categories).map(([cat, count]) => `
                            <tr>
                                <td>${cat}</td>
                                <td>${count}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            topCatsDiv.style.display = 'block';
        } else {
            const topCatsDiv = document.getElementById('top-categories');
            topCatsDiv.style.display = 'none';
            topCatsDiv.innerHTML = '';
        }
        
        // Update related features
        if (data.top_related && data.top_related.length > 0) {
            const relatedDiv = document.getElementById('related-features');
            relatedDiv.innerHTML = `
                <h3 style="font-size:10px; margin-bottom:4px; color:#555; font-weight:600;">Related features</h3>
                <table class="category-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Score</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.top_related.map(item => `
                            <tr>
                                <td>${item.feature}</td>
                                <td>${item.score.toFixed(3)}</td>
                                <td>${item.type}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            relatedDiv.style.display = 'block';
        } else {
            document.getElementById('related-features').style.display = 'none';
        }
        
        document.getElementById('stats-section').style.display = 'block';
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('plot1-container').style.display = 'none';
        document.getElementById('plot2-container').style.display = 'none';
        document.getElementById('plot2-stats').style.display = 'none';
        document.getElementById('stats-section').style.display = 'none';
        document.getElementById('left-species-counter').style.display = 'none';
        document.getElementById('empty-state').textContent = `Error: ${error.message}`;
        document.getElementById('empty-state').style.display = 'flex';
    } finally {
        document.getElementById('loading').classList.remove('active');
    }
}

function executePlotlyScripts(container, onDone) {
    const scripts = container.querySelectorAll('script');
    scripts.forEach(script => {
        try {
            if (script.src) {
                const newScript = document.createElement('script');
                newScript.src = script.src;
                document.head.appendChild(newScript);
            } else {
                eval(script.textContent);
            }
        } catch (err) {
            console.error('Error executing data plot script:', err);
        }
    });
    if (typeof onDone === 'function') {
        requestAnimationFrame(() => setTimeout(onDone, 60));
    }
}