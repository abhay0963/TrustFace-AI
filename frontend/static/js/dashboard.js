function decisionBadge(decision) {
    const map = {
        auto_accept: ["badge-accept", "AUTO ACCEPT"],
        retry: ["badge-retry", "RETRY"],
        reject: ["badge-reject", "REJECT"],
        unknown: ["badge-unknown", "UNKNOWN"],
    };
    const [cls, label] = map[decision] || ["badge-unknown", decision.toUpperCase()];
    return `<span class="badge ${cls}">${label}</span>`;
}

async function loadDashboard() {
    const res = await fetch("/api/dashboard-summary");
    const data = await res.json();

    document.getElementById("stat-total-users").textContent = data.total_registered_users;
    document.getElementById("stat-present").textContent = data.present_today;
    document.getElementById("stat-trust").textContent = data.avg_trust_score + "%";
    document.getElementById("stat-presence").textContent = data.avg_presence_percentage + "%";

    renderAttendanceTable(data.today_attendance);
    renderPresenceChart(data.today_attendance);
    renderRecentLogs(data.recent_logs);
}

function presenceCategoryBadge(category) {
    const map = {
        "Continuous Attendance": "badge-accept",
        "Frequent Exits": "badge-retry",
        "Brief Appearance": "badge-retry",
        "Suspicious Intermittent Presence": "badge-reject",
        "Interrupted Presence": "badge-unknown",
    };
    return `<span class="badge ${map[category] || "badge-unknown"}">${category}</span>`;
}

function renderAttendanceTable(rows) {
    const wrap = document.getElementById("attendance-table-wrap");
    if (!rows.length) {
        wrap.innerHTML = `<div class="empty-state"><div class="icon">◈</div>No attendance recorded yet today.<br>Head to <a href="/attendance" style="color: var(--accent);">Live Attendance</a> to begin.</div>`;
        return;
    }
    let html = `<table><thead><tr><th>Name</th><th>ID</th><th>Entry</th><th>Last Seen</th><th>Presence</th><th>Category</th><th>Trust</th></tr></thead><tbody>`;
    for (const r of rows) {
        html += `<tr>
            <td>${r.name}</td>
            <td class="mono tag-mono">${r.external_id}</td>
            <td class="mono">${r.entry_time_formatted}</td>
            <td class="mono">${r.last_seen_formatted}</td>
            <td class="mono">${r.presence_percentage}% <span class="tag-mono">(${r.presence_duration_formatted}, ${r.presence_consistency}% consistent)</span></td>
            <td>${presenceCategoryBadge(r.presence_category)}</td>
            <td class="mono">${r.avg_trust_score.toFixed(1)}</td>
        </tr>`;
    }
    html += `</tbody></table>`;
    wrap.innerHTML = html;
}

function renderPresenceChart(rows) {
    const el = document.getElementById("presence-chart");
    if (!rows.length) {
        el.innerHTML = `<div class="empty-state" style="padding-top: 90px;">No data yet</div>`;
        return;
    }
    const names = rows.map(r => r.name);
    const pct = rows.map(r => r.presence_percentage);

    Plotly.newPlot(el, [{
        x: pct, y: names, type: "bar", orientation: "h",
        marker: { color: "#2dd4bf" },
    }], {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#8fa1b3", family: "IBM Plex Mono", size: 11 },
        margin: { l: 100, r: 20, t: 10, b: 30 },
        xaxis: { range: [0, 100], gridcolor: "#212b35", title: "%" },
        yaxis: { automargin: true },
    }, { displayModeBar: false, responsive: true });
}

function renderRecentLogs(logs) {
    const wrap = document.getElementById("recent-logs-wrap");
    if (!logs.length) {
        wrap.innerHTML = `<div class="empty-state"><div class="icon">▤</div>No recognition events yet.</div>`;
        return;
    }
    let html = `<table><thead><tr><th>Time</th><th>Name</th><th>Similarity</th><th>Trust</th><th>Decision</th></tr></thead><tbody>`;
    for (const l of logs) {
        html += `<tr>
            <td class="mono tag-mono">${new Date(l.timestamp).toLocaleTimeString()}</td>
            <td>${l.name || "Unknown"}</td>
            <td class="mono">${l.similarity !== null ? l.similarity.toFixed(3) : "—"}</td>
            <td class="mono">${l.trust_score !== null ? l.trust_score.toFixed(1) : "—"}</td>
            <td>${decisionBadge(l.decision)}</td>
        </tr>`;
    }
    html += `</tbody></table>`;
    wrap.innerHTML = html;
}

loadDashboard();
setInterval(loadDashboard, 5000);
