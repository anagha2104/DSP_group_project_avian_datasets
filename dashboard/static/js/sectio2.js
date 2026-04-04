currentFilters = {};
const PLOT_CONFIG = { responsive: true, displayModeBar: false };

function mergeTightMargins(layout, overrides = {}) {
    const base = { l: 36, r: 18, t: 40, b: 36, pad: 0 };
    const m = layout && layout.margin ? { ...base, ...layout.margin, ...overrides } : { ...base, ...overrides };
    return { ...(layout || {}), margin: m };
}

function filtersWithGeo() {
    const latMin = parseInt(document.getElementById("lat-min").value, 10);
    const latMax = parseInt(document.getElementById("lat-max").value, 10);
    const lonMin = parseInt(document.getElementById("lon-min").value, 10);
    const lonMax = parseInt(document.getElementById("lon-max").value, 10);
    const f = { ...currentFilters };
    if (latMin > -72 || latMax < 79) f.lat_range = [latMin, latMax];
    if (lonMin > -179 || lonMax < 180) f.lon_range = [lonMin, lonMax];
    return f;
}

const filterCols = {
    order_birdlife: "Order",
    family_birdlife: "Family",
    trophic_niche: "Trophic Niche",
    habitat_density: "Habitat Density",
};

function isNum(col) {
    return NUMERICAL_TRAITS.includes(col);
}

function setPlotModeOptions() {
    const x = document.getElementById("feature-x").value;
    const y = document.getElementById("feature-y").value;
    const plotMode = document.getElementById("plot-mode");
    const numNumControls = document.getElementById("num-num-controls");

    plotMode.innerHTML = "";
    if (!x || !y) {
        numNumControls.style.display = "none";
        return;
    }

    if (isNum(x) && isNum(y)) {
        plotMode.innerHTML = `<option value="scatter_regression">Scatter + Regression</option>`;
        numNumControls.style.display = "block";
    } else if (isNum(x) !== isNum(y)) {
        plotMode.innerHTML = `
            <option value="bar">Bar (Mean)</option>
            <option value="barh_stacked">Stacked Bar (H)</option>
        `;
        numNumControls.style.display = "none";
    } else {
        plotMode.innerHTML = `<option value="heatmap">Heatmap</option><option value="grouped_bar">Grouped Bar</option>`;
        numNumControls.style.display = "none";
    }
}

function getNormalizedPlotMode(x, y, selectedMode) {
    if (isNum(x) && isNum(y)) return "scatter_regression";
    if (isNum(x) !== isNum(y)) {
        if (selectedMode === "barh_stacked") return "barh_stacked";
        return "bar";
    }
    return selectedMode === "grouped_bar" ? "grouped_bar" : "heatmap";
}

function setConclusionPanelVisible(visible, html = "") {
    const panel = document.getElementById("conclusion-panel");
    const body = document.getElementById("conclusion");
    if (!panel || !body) return;
    if (!visible) {
        panel.classList.remove("is-visible");
        body.innerHTML = "";
        return;
    }
    body.innerHTML = html;
    panel.classList.add("is-visible");
}

function renderSubgroupToggle(stats, plotElement) {
    const box = document.getElementById("subgroup-toggle-container");
    if (!box) return;
    const values = stats?.subfeature_values || [];
    const name = stats?.subfeature_name;
    if (!name || !values.length || !plotElement?.data?.length) {
        box.style.display = "none";
        box.innerHTML = "";
        return;
    }

    box.innerHTML = `
        <h3 class="plot-below-title">Show ${name} groups</h3>
        <div class="subgroup-toggles-wrap" id="subgroup-checks">
            <label><input type="checkbox" id="subgroup-all" checked /> All</label>
            ${values.map(v => `<label><input type="checkbox" class="subgroup-item" value="${v}" checked /> ${v}</label>`).join("")}
        </div>
    `;
    box.style.display = "block";

    const getSelected = () => new Set(Array.from(document.querySelectorAll(".subgroup-item:checked")).map(x => x.value));

    const renderSubfeatureStats = (selectedSet) => {
        const subgroupStats = stats?.subfeature_stats || {};
        const detailRows = Object.entries(subgroupStats)
            .filter(([k]) => selectedSet.has(k))
            .map(([k, v]) => [
                k, v.count,
                v.x_mean.toFixed(2), v.x_median.toFixed(2), v.x_std !== undefined ? v.x_std.toFixed(2) : "N/A",
                v.y_mean.toFixed(2), v.y_median.toFixed(2), v.y_std !== undefined ? v.y_std.toFixed(2) : "N/A",
            ]);
        renderDetailTable("Subfeature-wise stats", detailRows, ["Group", "Count", "X Mean", "X Med", "X Std", "Y Mean", "Y Med", "Y Std"]);
    };

    const applyVisibility = () => {
        const selected = getSelected();
        const allChecked = selected.size === values.length;
        document.getElementById("subgroup-all").checked = allChecked;
        plotElement.data.forEach((tr, i) => {
            const traceName = String(tr.name || "");
            const matchedGroup = values.find(v => traceName === v || traceName === `${v} regression`);
            if (matchedGroup) {
                Plotly.restyle(plotElement, { visible: selected.has(matchedGroup) ? true : "legendonly" }, [i]);
            } else {
                Plotly.restyle(plotElement, { visible: true }, [i]);
            }
        });
        renderSubfeatureStats(selected);
    };

    document.getElementById("subgroup-all").addEventListener("change", (e) => {
        document.querySelectorAll(".subgroup-item").forEach(c => c.checked = e.target.checked);
        applyVisibility();
    });
    document.querySelectorAll(".subgroup-item").forEach(c => c.addEventListener("change", applyVisibility));
    applyVisibility();
}

async function buildFilters() {
    const container = document.getElementById("filters-container");
    for (const [col, label] of Object.entries(filterCols)) {
        const wrapper = document.createElement("div");
        wrapper.className = "filter-row";
        wrapper.innerHTML = `<label>${label}:</label><select data-filter="${col}" class="filter-select"><option value="">All</option></select>`;
        container.appendChild(wrapper);

        const res = await fetch(`/api/filter-values/${col}`);
        const data = await res.json();
        const select = wrapper.querySelector(".filter-select");
        (data.values || []).forEach((v) => {
            const op = document.createElement("option");
            op.value = v;
            op.textContent = v;
            select.appendChild(op);
        });
        select.addEventListener("change", () => {
            if (select.value) currentFilters[col] = select.value;
            else delete currentFilters[col];
            updateSection2();
        });
    }
}

function clearAllFilters() {
    currentFilters = {};
    document.querySelectorAll(".filter-select").forEach((s) => (s.value = ""));
    document.getElementById("lat-min").value = -72;
    document.getElementById("lat-max").value = 79;
    document.getElementById("lon-min").value = -179;
    document.getElementById("lon-max").value = 180;
    document.getElementById("lat-min-val").textContent = "-72";
    document.getElementById("lat-max-val").textContent = "79";
    document.getElementById("lon-min-val").textContent = "-179";
    document.getElementById("lon-max-val").textContent = "180";
    updateSection2();
}

function renderDetailTable(title, rows, headers) {
    const detail = document.getElementById("detail-table");
    if (!rows || !rows.length) {
        detail.style.display = "none";
        detail.innerHTML = "";
        return;
    }
    detail.innerHTML = `
        <h3 style="font-size:10px; margin-bottom:4px; color:#555; font-weight:600;">${title}</h3>
        <table class="category-table">
            <thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
            <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
    `;
    detail.style.display = "block";
}

async function updateSection2() {
    const x = document.getElementById("feature-x").value;
    const y = document.getElementById("feature-y").value;
    setPlotModeOptions();
    if (!x || !y) {
        document.getElementById("plot-container").style.display = "none";
        document.getElementById("stats-section").style.display = "none";
        document.getElementById("left-species-counter").style.display = "none";
        setConclusionPanelVisible(false);
        document.getElementById("empty-state").textContent = "Select Feature X and Feature Y to start";
        document.getElementById("empty-state").style.display = "flex";
        return;
    }
    if (x === y) {
        document.getElementById("plot-container").style.display = "none";
        document.getElementById("stats-section").style.display = "none";
        document.getElementById("left-species-counter").style.display = "none";
        setConclusionPanelVisible(false);
        document.getElementById("empty-state").textContent = "Feature X and Feature Y must be different.";
        document.getElementById("empty-state").style.display = "flex";
        return;
    }

    const selectedMode = document.getElementById("plot-mode").value;
    const normalizedMode = getNormalizedPlotMode(x, y, selectedMode);
    document.getElementById("plot-mode").value = normalizedMode;

    const payload = {
        feature_x: x,
        feature_y: y,
        subfeature: document.getElementById("subfeature-select").value || null,
        filters: filtersWithGeo(),
        plot_mode: normalizedMode,
        show_ci: document.getElementById("show-ci").checked,
        log_x: document.getElementById("log-x").checked,
        log_y: document.getElementById("log-y").checked,
    };

    document.getElementById("loading").classList.add("active");
    document.getElementById("empty-state").style.display = "none";
    try {
        const res = await fetch("/api/analysis2", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Section 2 analysis failed");

        document.getElementById("left-filtered-count").textContent = data.filtered_count;
        document.getElementById("left-total-count").textContent = data.total_count;
        document.getElementById("left-species-counter").style.display = "block";

        if (data.no_data) {
            document.getElementById("plot-container").style.display = "none";
            document.getElementById("stats-section").style.display = "none";
            setConclusionPanelVisible(false);
            const eqEl = document.getElementById("equation-display");
            if (eqEl) {
                eqEl.style.display = "none";
                eqEl.innerHTML = "";
            }
            const subgroupEl = document.getElementById("subgroup-toggle-container");
            if (subgroupEl) {
                subgroupEl.style.display = "none";
                subgroupEl.innerHTML = "";
            }
            document.getElementById("empty-state").textContent = data.message || "There is no species for selected filters.";
            document.getElementById("empty-state").style.display = "flex";
            return;
        }

        if (!data.plot || !data.plot.data) {
            throw new Error("Plot payload missing from analysis response");
        }

        const plot = document.getElementById("plot-container");
        plot.innerHTML = "";
        plot.style.display = "block";
        Plotly.newPlot(plot, data.plot.data, mergeTightMargins(data.plot.layout || {}), PLOT_CONFIG);
        const eqBox = document.getElementById("equation-display");
        const equations = data?.stats?.regression_equations || {};
        const eqEntries = Object.entries(equations);
        if (eqBox && eqEntries.length > 0) {
            eqBox.innerHTML = `
                <h3 class="plot-below-title">Regression equation${eqEntries.length > 1 ? "s" : ""}</h3>
                <div class="equation-lines equation-lines-row">
                    ${eqEntries.map(([k, v]) => `<div class="equation-item"><strong>${k}</strong>: y = ${Number(v.slope).toFixed(4)}x + ${Number(v.intercept).toFixed(4)}</div>`).join("")}
                </div>
            `;
            eqBox.style.display = "block";
        } else if (eqBox) {
            eqBox.style.display = "none";
            eqBox.innerHTML = "";
        }

        const stats = data.stats || {};
        const statsDisplay = document.getElementById("stats-display");
        const rows = [];
        rows.push(["Sample size", stats.sample_size ?? "N/A"]);
        rows.push(["Method", stats.method ?? "N/A"]);
        if (typeof stats.pearson_r === "number" && typeof stats.pearson_p === "number") {
            rows.push(["Pearson r (p)", `${stats.pearson_r.toFixed(3)} (${stats.pearson_p.toExponential(2)})`]);
        }
        if (typeof stats.spearman_r === "number" && typeof stats.spearman_p === "number") {
            rows.push(["Spearman r (p)", `${stats.spearman_r.toFixed(3)} (${stats.spearman_p.toExponential(2)})`]);
        }
        if (typeof stats.eta_squared === "number") rows.push(["Eta-squared", stats.eta_squared.toFixed(3)]);
        if (typeof stats.cramers_v === "number") rows.push(["Cramer's V", stats.cramers_v.toFixed(3)]);
        if (stats.p_value !== undefined && stats.p_value !== null) rows.push(["p-value", Number(stats.p_value).toExponential(2)]);

        const fx = stats.feature_x_summary || {};
        const fy = stats.feature_y_summary || {};
        rows.push(["Feature X mean/median/std", `${fx.mean ?? "N/A"} / ${fx.median ?? "N/A"} / ${fx.std ?? "N/A"}`]);
        rows.push(["Feature Y mean/median/std", `${fy.mean ?? "N/A"} / ${fy.median ?? "N/A"} / ${fy.std ?? "N/A"}`]);

        statsDisplay.innerHTML = rows.map((r) => `<div class="stat-row"><span>${r[0]}</span><span>${r[1]}</span></div>`).join("");

        if (stats.category_stats) {
            const detailRows = Object.entries(stats.category_stats).map(([k, v]) => [k, v.count, v.mean.toFixed(2), v.median.toFixed(2)]);
            renderDetailTable("Per-category stats", detailRows, ["Category", "Count", "Mean", "Median"]);
        } else if (stats.frequency_tables) {
            const freqX = Object.entries(stats.frequency_tables.feature_x_frequency || {}).map(([k, v]) => [`X: ${k}`, v]);
            const freqY = Object.entries(stats.frequency_tables.feature_y_frequency || {}).map(([k, v]) => [`Y: ${k}`, v]);
            renderDetailTable("Frequency tables", [...freqX, ...freqY], ["Group", "Count"]);
        } else {
            renderDetailTable("", [], []);
        }
        renderSubgroupToggle(stats, plot);

        const boldify = (t) => (t || "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        if (data.conclusion) {
            setConclusionPanelVisible(true, boldify(data.conclusion));
        } else {
            setConclusionPanelVisible(false);
        }
        document.getElementById("stats-section").style.display = "block";
    } catch (e) {
        document.getElementById("plot-container").style.display = "none";
        document.getElementById("stats-section").style.display = "none";
        document.getElementById("left-species-counter").style.display = "none";
        setConclusionPanelVisible(false);
        const eqEl = document.getElementById("equation-display");
        if (eqEl) {
            eqEl.style.display = "none";
            eqEl.innerHTML = "";
        }
        const subgroupEl = document.getElementById("subgroup-toggle-container");
        if (subgroupEl) {
            subgroupEl.style.display = "none";
            subgroupEl.innerHTML = "";
        }
        document.getElementById("empty-state").textContent = `Error: ${e.message}`;
        document.getElementById("empty-state").style.display = "flex";
    } finally {
        document.getElementById("loading").classList.remove("active");
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    await buildFilters();
    setPlotModeOptions();
    ["feature-x", "feature-y"].forEach((id) => {
        document.getElementById(id).addEventListener("change", () => {
            setPlotModeOptions();
            updateSection2();
        });
    });
    ["subfeature-select", "plot-mode", "show-ci", "log-x", "log-y"].forEach((id) => {
        document.getElementById(id).addEventListener("change", updateSection2);
    });
    ["lat-min", "lat-max", "lon-min", "lon-max"].forEach((id) => {
        const el = document.getElementById(id);
        el.addEventListener("input", () => {
            document.getElementById(`${id}-val`).textContent = el.value;
        });
        el.addEventListener("change", updateSection2);
    });
});

