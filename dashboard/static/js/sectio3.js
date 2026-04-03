let selectedSpecies = [];
let lastResponse = null;
let menuState = { query: "", offset: 0, total: 0, items: [], activeIndex: -1, loading: false };

/** Which trait columns to show in Row stats (exactly 5 keys you care about). Edit anytime. */
const SECTION3_ROW_TRAIT_KEYS = ["avibase_id", "wing_len", "tarsus", "hwi", "beak_depth", "mass","tail","kipps","secondary"]

const PLOT_CONFIG = { responsive: true, displayModeBar: false };

const MAP_PLOT_CONFIG = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
    doubleClick: "reset",
};

function formatPhyloDistance(v) {
    if (v === null || v === undefined) return "—";
    const x = Number(v);
    if (!Number.isFinite(x)) return "—";
    if (x >= 100) return x.toFixed(1);
    if (x >= 10) return x.toFixed(2);
    if (x >= 1) return x.toFixed(3);
    return x.toFixed(4);
}

function section3PlotHeight(el) {
    if (!el) return Math.max(260, Math.floor(window.innerHeight * 0.52));
    const r = el.getBoundingClientRect();
    const h = Math.floor(r.height);
    return Math.max(220, h > 50 ? h : Math.floor(window.innerHeight * 0.52));
}

function resizeS3Plots() {
    ["radar", "map"].forEach((id) => {
        const el = document.getElementById(id);
        if (!el || el.style.display === "none") return;
        const gd = el.querySelector(".js-plotly-plot");
        if (gd) Plotly.Plots.resize(gd);
    });
}

async function fetchSpeciesSuggestions(query, offset = 0, limit = 60) {
    const res = await fetch(`/api/species-search?q=${encodeURIComponent(query)}&offset=${offset}&limit=${limit}`);
    const data = await res.json();
    return data;
}

function renderBadges() {
    document.getElementById("selected-count").textContent = selectedSpecies.length;
    const holder = document.getElementById("selected-badges");
    holder.innerHTML = selectedSpecies
        .map(
            (s) =>
                `<span class="badge" onclick="removeSpecies('${s.replace(/'/g, "\\'")}')">${s}<span class="remove">×</span></span>`
        )
        .join("");
}

function addSpecies() {
    const input = document.getElementById("species-input");
    const value = (input.value || "").trim();
    if (!value) return;
    if (!selectedSpecies.includes(value)) selectedSpecies.push(value);
    input.value = "";
    hideSpeciesMenu();
    renderBadges();
    updateComparison();
}

function removeSpecies(name) {
    selectedSpecies = selectedSpecies.filter((s) => s !== name);
    renderBadges();
    updateComparison();
}

function clearSpecies() {
    selectedSpecies = [];
    lastResponse = null;
    renderBadges();
    document.getElementById("rows-container").innerHTML = "";
    document.getElementById("distances").innerHTML = "";
    document.getElementById("radar").style.display = "none";
    document.getElementById("map").style.display = "none";
    document.getElementById("map-missing").style.display = "none";
    document.getElementById("conclusion").classList.remove("is-open");
    document.getElementById("conclusion").innerHTML = "";
}

function rowTraitEntries(rowObj) {
    const pairs = [];
    if (Object.prototype.hasOwnProperty.call(rowObj, "species_birdtree")) {
        pairs.push(["species_birdtree", rowObj.species_birdtree]);
    }
    for (const k of SECTION3_ROW_TRAIT_KEYS) {
        if (Object.prototype.hasOwnProperty.call(rowObj, k)) {
            pairs.push([k, rowObj[k]]);
        }
    }
    return pairs;
}

function renderRowTables(rows) {
    const container = document.getElementById("rows-container");
    if (!rows || rows.length === 0) {
        container.innerHTML = `<div style="color:#999; font-size:10px;">Select species to view traits.</div>`;
        return;
    }

    container.innerHTML = rows
        .map((rowObj) => {
            const name = rowObj.species_birdtree || "Unknown";
            const pairs = rowTraitEntries(rowObj);
            const body = pairs
                .map(([k, v]) => `<tr><td>${k}</td><td>${v === null || v === undefined ? "" : v}</td></tr>`)
                .join("");
            return `
                <h3 class="mini" style="margin-top:6px;">${name}</h3>
                <table class="category-table">
                    <thead><tr><th>Trait</th><th>Value</th></tr></thead>
                    <tbody>${body}</tbody>
                </table>
            `;
        })
        .join("");
}

function renderDistanceTable(distanceMatrix) {
    const div = document.getElementById("distances");
    if (!distanceMatrix || distanceMatrix.length === 0) {
        div.innerHTML = "";
        return;
    }
    const header = `<tr>
        <th>Species pair</th>
        <th title="Euclidean distance on Z-scored overlapping numeric traits">Morphology (norm.)</th>
        <th title="Number of numeric traits used for morphology distance">n traits</th>
        <th class="dist-phylo-head" title="Path length on the phylogenetic tree (same units as Newick branch lengths)">Phylogeny</th>
    </tr>`;
    const rows = distanceMatrix
        .map((d) => {
            const morph =
                d.euclidean === null || d.euclidean === undefined ? "—" : Number(d.euclidean).toFixed(3);
            return `<tr>
                <td class="dist-pair">${d.a} ↔ ${d.b}</td>
                <td class="dist-morph">${morph}</td>
                <td class="dist-n">${d.euclidean_n ?? "—"}</td>
                <td class="dist-phylo">${formatPhyloDistance(d.phylo)}</td>
            </tr>`;
        })
        .join("");
    div.innerHTML = `
        <h3 class="mini">Pairwise distances</h3>
        <p class="dist-legend">Morphology = standardized trait space. Phylogeny = tree distance (— if a tip is missing from the tree).</p>
        <table class="category-table distance-table"><thead>${header}</thead><tbody>${rows}</tbody></table>
    `;
}

function renderRadar(radarFeatures, radarSeries) {
    const el = document.getElementById("radar");
    if (!radarSeries || radarSeries.length < 2) {
        el.style.display = "none";
        el.innerHTML = "";
        return;
    }
    el.style.display = "block";
    const palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"];
    const traces = radarSeries.map((s, idx) => ({
        type: "scatterpolar",
        r: s.values.map((v) => (v === null ? null : v)),
        theta: radarFeatures,
        fill: "toself",
        name: s.name,
        opacity: 0.28,
        line: { width: 2, color: palette[idx % palette.length] },
        marker: { size: 3, color: palette[idx % palette.length] },
    }));
    const runPlot = () => {
        const h = section3PlotHeight(el);
        Plotly.newPlot(
            el,
            traces,
            {
                margin: { l: 28, r: 28, t: 28, b: 28, pad: 0 },
                title: { text: "Trait radar", font: { size: 11, color: "#334155" } },
                polar: {
                    bgcolor: "rgba(248,250,252,0.9)",
                    radialaxis: {
                        visible: true,
                        gridshape: "circular",
                        gridcolor: "#cbd5e1",
                        gridwidth: 0.6,
                        tickfont: { size: 8, color: "#64748b" },
                    },
                    angularaxis: {
                        gridcolor: "#cbd5e1",
                        tickfont: { size: 9, color: "#334155" },
                    },
                },
                paper_bgcolor: "rgba(0,0,0,0)",
                showlegend: true,
                legend: {
                    orientation: "h",
                    xref: "paper",
                    yref: "paper",
                    x: 0.5,
                    y: 1.02,
                    xanchor: "center",
                    yanchor: "bottom",
                    font: { size: 9 },
                    bgcolor: "rgba(255,255,255,0.92)",
                },
                height: h,
                autosize: true,
            },
            PLOT_CONFIG
        );
        requestAnimationFrame(resizeS3Plots);
    };
    requestAnimationFrame(() => requestAnimationFrame(runPlot));
}

function renderMap(geoPoints, geoMissing) {
    const el = document.getElementById("map");
    const missingEl = document.getElementById("map-missing");
    if (!geoPoints || geoPoints.length === 0) {
        el.style.display = "none";
        el.innerHTML = "";
        if (geoMissing && geoMissing.length > 0) {
            missingEl.textContent = `⚠️ Geo not shown for: ${geoMissing.join(", ")} (lat/lon not in dataset)`;
            missingEl.style.display = "block";
        } else {
            missingEl.style.display = "none";
            missingEl.textContent = "";
        }
        return;
    }
    el.style.display = "block";

    // Same palette / style as Section 1 species map (markers only, full globe, legend bottom)
    const PALETTE = [
        "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#F97316", "#EC4899", "#84CC16", "#6366F1",
        "#14B8A6", "#A855F7", "#F43F5E", "#22C55E", "#FACC15", "#64748B", "#0EA5E9", "#D946EF", "#FB923C", "#2DD4BF",
    ];

    const traces = geoPoints.map((p, i) => {
        const label = p.name.split("_").join(" ");
        return {
            type: "scattergeo",
            mode: "markers",
            name: label,
            lat: [p.lat],
            lon: [p.lon],
            text: [p.name],
            marker: {
                size: 6,
                color: PALETTE[i % PALETTE.length],
                opacity: 0.82,
                line: { width: 0.5, color: "white" },
            },
            hovertemplate: `<b>${label}</b><br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>`,
        };
    });

    const runMap = () => {
        const h = section3PlotHeight(el);
        Plotly.newPlot(
            el,
            traces,
            {
                margin: { l: 0, r: 0, t: 0, b: 0, pad: 0 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                showlegend: traces.length > 0,
                legend: {
                    orientation: "h",
                    xref: "paper",
                    yref: "paper",
                    x: 0.5,
                    y: 0.012,
                    xanchor: "center",
                    yanchor: "bottom",
                    font: { size: 7, family: "Segoe UI, sans-serif" },
                    bgcolor: "rgba(255,255,255,0.96)",
                    bordercolor: "rgba(180,180,180,0.4)",
                    borderwidth: 1,
                    itemsizing: "constant",
                    itemwidth: 14,
                    tracegroupgap: 4,
                    traceorder: "normal",
                },
                geo: {
                    showland: true,
                    landcolor: "#f0f0f0",
                    coastlinecolor: "#ccc",
                    projection: { type: "natural earth" },
                    bgcolor: "rgba(0,0,0,0)",
                    domain: { x: [0, 1], y: [0, 1] },
                    lataxis: { range: [-90, 90], showgrid: true, gridcolor: "rgba(0,0,0,0.06)" },
                    lonaxis: { range: [-180, 180], showgrid: true, gridcolor: "rgba(0,0,0,0.06)" },
                },
                height: h,
                autosize: true,
            },
            MAP_PLOT_CONFIG
        );
        requestAnimationFrame(resizeS3Plots);
    };
    requestAnimationFrame(() => requestAnimationFrame(runMap));

    if (geoMissing && geoMissing.length > 0) {
        missingEl.textContent = `⚠️ Geo not shown for: ${geoMissing.join(", ")} (lat/lon not in dataset)`;
        missingEl.style.display = "block";
    } else {
        missingEl.style.display = "none";
        missingEl.textContent = "";
    }
}

function showSpeciesMenu() {
    const menu = document.getElementById("species-menu");
    menu.style.display = "block";
}

function hideSpeciesMenu() {
    const menu = document.getElementById("species-menu");
    menu.style.display = "none";
    menuState.activeIndex = -1;
}

function renderSpeciesMenu() {
    const menu = document.getElementById("species-menu");
    if (!menuState.items.length && !menuState.loading) {
        menu.innerHTML = `<div class="dropdown-item" style="color:#777;">No matches</div>`;
        return;
    }
    menu.innerHTML = menuState.items
        .map((s, idx) => {
            const cls = idx === menuState.activeIndex ? "dropdown-item active" : "dropdown-item";
            return `<div class="${cls}" data-idx="${idx}">${s}</div>`;
        })
        .join("");
    if (menuState.loading) {
        menu.innerHTML += `<div class="dropdown-item" style="color:#777;">Loading…</div>`;
    } else if (menuState.items.length < menuState.total) {
        menu.innerHTML += `<div class="dropdown-item" style="color:#007bff;">Scroll to load more…</div>`;
    }

    menu.querySelectorAll(".dropdown-item[data-idx]").forEach((el) => {
        el.addEventListener("mousedown", (e) => {
            e.preventDefault();
            const idx = Number(el.dataset.idx);
            const chosen = menuState.items[idx];
            document.getElementById("species-input").value = chosen;
            hideSpeciesMenu();
        });
    });
}

async function refreshMenu(query) {
    menuState.query = query;
    menuState.offset = 0;
    menuState.items = [];
    menuState.total = 0;
    menuState.activeIndex = -1;
    await loadMoreMenu();
}

async function loadMoreMenu() {
    if (menuState.loading) return;
    menuState.loading = true;
    showSpeciesMenu();
    renderSpeciesMenu();
    const data = await fetchSpeciesSuggestions(menuState.query, menuState.offset, 60);
    const values = data.values || [];
    menuState.total = data.total || values.length;
    menuState.items = [...menuState.items, ...values];
    menuState.offset += values.length;
    menuState.loading = false;
    renderSpeciesMenu();
}

async function updateComparison() {
    const loading = document.getElementById("loading");
    loading.classList.add("active");
    try {
        const res = await fetch("/api/analysis3", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ species: selectedSpecies }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Section 3 failed");
        lastResponse = data;
        renderRowTables(data.rows || []);
        renderDistanceTable(data.distances || []);
        renderRadar(data.radar_features || [], data.radar_series || []);
        renderMap(data.geo_points || [], data.geo_missing || []);

        const concl = document.getElementById("conclusion");
        if (data.conclusion) {
            const boldify = (t) => (t || "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
            concl.innerHTML = `<div class="conclusion-heading">Conclusion</div><div class="conclusion-inner">${boldify(data.conclusion)}</div>`;
            concl.classList.add("is-open");
        } else {
            concl.classList.remove("is-open");
            concl.innerHTML = "";
        }
    } catch (e) {
        document.getElementById("rows-container").innerHTML = `<div style="color:#b00; font-size:10px;">Error: ${e.message}</div>`;
        document.getElementById("distances").innerHTML = "";
        document.getElementById("radar").style.display = "none";
        document.getElementById("map").style.display = "none";
    } finally {
        loading.classList.remove("active");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    renderBadges();

    let resizeT;
    window.addEventListener("resize", () => {
        clearTimeout(resizeT);
        resizeT = setTimeout(resizeS3Plots, 120);
    });

    const input = document.getElementById("species-input");
    const menu = document.getElementById("species-menu");
    let debounce = null;

    input.addEventListener("focus", async () => {
        if (!menuState.items.length) await refreshMenu("");
        showSpeciesMenu();
    });

    input.addEventListener("input", () => {
        const q = input.value.trim();
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(async () => {
            if (q.length === 0) {
                await refreshMenu("");
                return;
            }
            if (q.length < 2) return;
            await refreshMenu(q);
        }, 220);
    });

    menu.addEventListener("scroll", async () => {
        const nearBottom = menu.scrollTop + menu.clientHeight >= menu.scrollHeight - 30;
        if (nearBottom && menuState.items.length < menuState.total) {
            await loadMoreMenu();
        }
    });

    input.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            menuState.activeIndex = Math.min(menuState.activeIndex + 1, menuState.items.length - 1);
            renderSpeciesMenu();
            return;
        }
        if (e.key === "ArrowUp") {
            e.preventDefault();
            menuState.activeIndex = Math.max(menuState.activeIndex - 1, 0);
            renderSpeciesMenu();
            return;
        }
        if (e.key === "Enter") {
            e.preventDefault();
            if (menuState.activeIndex >= 0 && menuState.items[menuState.activeIndex]) {
                input.value = menuState.items[menuState.activeIndex];
                hideSpeciesMenu();
            }
            addSpecies();
        }
        if (e.key === "Escape") {
            hideSpeciesMenu();
        }
    });

    document.addEventListener("click", (e) => {
        const within = e.target.closest(".dropdown");
        if (!within) hideSpeciesMenu();
    });
});

