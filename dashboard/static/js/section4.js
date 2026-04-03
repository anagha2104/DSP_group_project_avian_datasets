let featureOptions = { numerical: [], categorical: [], all: [] };

function buildFeatureSelectHTML(selected = "") {
    const opts = featureOptions.all
        .map((f) => `<option value="${f}" ${selected === f ? "selected" : ""}>${f}</option>`)
        .join("");
    return `<select class="trait-feature"><option value="">-- feature --</option>${opts}</select>`;
}

function addTraitRow(feature = "", value = "") {
    const container = document.getElementById("traits-container");
    const row = document.createElement("div");
    row.className = "form-row";
    row.innerHTML = `
        ${buildFeatureSelectHTML(feature)}
        <input class="trait-value" type="text" value="${value}" placeholder="value" />
        <button class="btn-danger" type="button">×</button>
    `;
    row.querySelector("button").addEventListener("click", () => row.remove());
    container.appendChild(row);
}

function clearTraits() {
    document.getElementById("traits-container").innerHTML = "";
    addTraitRow();
    document.getElementById("result-table").innerHTML = "";
    document.getElementById("conclusion").style.display = "none";
}

function collectTraitInputs() {
    return Array.from(document.querySelectorAll("#traits-container .form-row")).map((row) => ({
        feature: row.querySelector(".trait-feature").value,
        value: row.querySelector(".trait-value").value,
    }));
}

async function runPrediction() {
    const loading = document.getElementById("loading");
    loading.classList.add("active");
    try {
        const payload = {
            traits: collectTraitInputs(),
            target_feature: document.getElementById("target-feature").value || null,
            top_k: Number(document.getElementById("top-k").value || 10),
        };

        const res = await fetch("/api/analysis4", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Prediction failed");

        renderResults(data.results || [], data.target_feature || null);
        const c = document.getElementById("conclusion");
        c.textContent = data.conclusion || "";
        c.style.display = data.conclusion ? "block" : "none";
    } catch (e) {
        document.getElementById("result-table").innerHTML = `<div style="color:#5c3d3d; font-size:11px;">Error: ${e.message}</div>`;
    } finally {
        loading.classList.remove("active");
    }
}

function renderResults(results, targetFeature) {
    const holder = document.getElementById("result-table");
    if (!results.length) {
        holder.innerHTML = `<div style="color:#4f4f4f; font-size:11px;">No matching species found for the entered traits.</div>`;
        return;
    }
    const headers = ["Rank", "Species", "Distance", "Numeric features used"];
    if (targetFeature) headers.push(targetFeature);
    const head = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
    const rows = results
        .map((r, idx) => {
            const cells = [
                String(idx + 1),
                r.species_birdtree,
                typeof r.distance === "number" ? r.distance.toFixed(3) : "N/A",
                String(r.numeric_features_used ?? 0),
            ];
            if (targetFeature) cells.push(r.target_feature_value ?? "");
            return `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
        })
        .join("");
    holder.innerHTML = `<table class="category-table"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

async function init() {
    const res = await fetch("/api/feature-options");
    const data = await res.json();
    featureOptions = data;

    const target = document.getElementById("target-feature");
    target.innerHTML =
        `<option value="">-- None --</option>` +
        featureOptions.all.map((f) => `<option value="${f}">${f}</option>`).join("");

    addTraitRow();
}

document.addEventListener("DOMContentLoaded", init);

