function decisionBadge(decision) {
    const map = {
        auto_accept: ["badge-accept", "ACCEPT"],
        retry: ["badge-retry", "RETRY"],
        reject: ["badge-reject", "REJECT"],
        unknown: ["badge-unknown", "UNKNOWN"],
        accept: ["badge-accept", "ACCEPT"],
    };
    const [cls, label] = map[decision] || ["badge-unknown", (decision || "—").toUpperCase()];
    return `<span class="badge ${cls}">${label}</span>`;
}

async function loadMlStatus() {
    const res = await fetch("/api/ml-status");
    const data = await res.json();

    document.getElementById("ml-labeled-count").textContent = data.labeled_examples;
    document.getElementById("ml-min-required").textContent = data.min_required;
    document.getElementById("ml-model-status").textContent = data.model_exists ? "Yes" : "Not yet";

    const btn = document.getElementById("btn-train");
    if (!data.ready_to_train) {
        btn.disabled = true;
        document.getElementById("train-status-text").textContent =
            `Need ${data.min_required - data.labeled_examples} more labeled examples`;
    } else {
        btn.disabled = false;
        document.getElementById("train-status-text").textContent = "Ready to train";
    }

    if (data.metrics) {
        document.getElementById("ml-accuracy").textContent = (data.metrics.accuracy * 100).toFixed(1) + "%";
        renderMetricsDetail(data.metrics);
    }
}

function renderMetricsDetail(metrics) {
    document.getElementById("ml-metrics-detail").style.display = "block";

    let table = `<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>`;
    for (const cls of Object.keys(metrics.per_class_metrics)) {
        const m = metrics.per_class_metrics[cls];
        table += `<tr><td>${decisionBadge(cls)}</td><td class="mono">${m.precision}</td><td class="mono">${m.recall}</td><td class="mono">${m.f1}</td><td class="mono">${m.support}</td></tr>`;
    }
    table += `</tbody></table>`;
    document.getElementById("per-class-table").innerHTML = table +
        `<p class="small-note" style="margin-top:10px;">Trained on ${metrics.trained_on} examples, tested on ${metrics.tested_on} held-out examples never seen during training. Confusion matrix labels: ${metrics.confusion_matrix_labels.join(", ")} → ${JSON.stringify(metrics.confusion_matrix)}</p>`;

    const fi = document.getElementById("feature-importances");
    if (metrics.feature_importances) {
        let html = "";
        const entries = Object.entries(metrics.feature_importances).sort((a, b) => b[1] - a[1]);
        for (const [name, val] of entries) {
            html += `<div class="subscore-row">
                <div class="subscore-name">${name}</div>
                <div class="subscore-bar-track"><div class="subscore-bar-fill" style="width:${(val * 100).toFixed(0)}%"></div></div>
                <div class="subscore-val">${(val * 100).toFixed(1)}%</div>
            </div>`;
        }
        fi.innerHTML = html;
    } else {
        fi.innerHTML = `<p class="small-note">Not available for this model type (Decision Tree doesn't expose ensemble-style importances the same way — use Random Forest for this view).</p>`;
    }
}

document.getElementById("btn-train").addEventListener("click", async () => {
    const btn = document.getElementById("btn-train");
    const modelType = document.getElementById("model-type-select").value;
    btn.disabled = true;
    btn.textContent = "Training…";

    try {
        const res = await fetch("/api/train-model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_type: modelType }),
        });
        const data = await res.json();
        if (data.success) {
            document.getElementById("train-status-text").textContent =
                `Trained! Accuracy: ${(data.metrics.accuracy * 100).toFixed(1)}%`;
            loadMlStatus();
        } else {
            document.getElementById("train-status-text").textContent = data.error;
        }
    } catch (err) {
        document.getElementById("train-status-text").textContent = "Error: " + err.message;
    } finally {
        btn.textContent = "Train Model";
        btn.disabled = false;
    }
});

async function labelLog(logId, label, btnEl) {
    const row = btnEl.closest("tr");
    await fetch(`/api/logs/${logId}/label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
    });
    row.querySelector(".label-cell").innerHTML = `<span class="badge badge-accept">Labeled: ${label}</span>`;
    loadMlStatus();
}

async function loadLogs() {
    const res = await fetch("/api/logs?limit=100");
    const logs = await res.json();
    const wrap = document.getElementById("logs-table-wrap");

    if (!logs.length) {
        wrap.innerHTML = `<div class="empty-state">No recognition events logged yet.</div>`;
        return;
    }

    let html = `<table><thead><tr>
        <th>Time</th><th>Name</th><th>Sim</th><th>Trust</th><th>Rule (raw)</th><th>ML (raw)</th><th>Final (acted on)</th><th>Ground Truth Label</th>
    </tr></thead><tbody>`;

    for (const l of logs) {
        html += `<tr>
            <td class="mono tag-mono">${new Date(l.timestamp).toLocaleTimeString()}</td>
            <td>${l.name || "Unknown"}</td>
            <td class="mono">${l.similarity !== null ? l.similarity.toFixed(3) : "—"}</td>
            <td class="mono">${l.trust_score !== null ? l.trust_score.toFixed(1) : "—"}</td>
            <td>${decisionBadge(l.decision)}</td>
            <td>${l.ml_decision ? decisionBadge(l.ml_decision) : '<span class="tag-mono">no model yet</span>'}</td>
            <td>${decisionBadge(l.final_decision || l.decision)}</td>
            <td class="label-cell">
                ${l.human_label
                    ? `<span class="badge badge-accept">Labeled: ${l.human_label}</span>`
                    : `<button class="btn btn-ghost btn-sm" onclick="labelLog(${l.id}, 'accept', this)">✓ Accept</button>
                       <button class="btn btn-ghost btn-sm" onclick="labelLog(${l.id}, 'retry', this)">↻ Retry</button>
                       <button class="btn btn-ghost btn-sm" onclick="labelLog(${l.id}, 'reject', this)">✗ Reject</button>`
                }
            </td>
        </tr>`;
    }
    html += `</tbody></table>`;
    wrap.innerHTML = html;
}

loadLogs();
loadMlStatus();
