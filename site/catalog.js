const TYPE_ORDER = ["recreational", "utility", "infrastructure"];
const TYPE_LABEL = {
  recreational: "RECREATIONAL",
  utility: "UTILITY",
  infrastructure: "INFRASTRUCTURE",
};

async function loadMachines() {
  const r = await fetch("../machines.json");
  if (!r.ok) throw new Error(`machines.json: HTTP ${r.status}`);
  return (await r.json()).machines;
}

function footerHTML() {
  return `
  <hr>
  <footer>
    <p align="center">
      <small>
        <em>
          afterthebubble.ai was more than 100€/year, I'm ok wasting my time, a little bit less so with my money.
        </em>
      </small>
    </p>
  </footer>
  `;
}

// ---- left frame: the catalog --------------------------------------------------
function renderNav(machines, el) {
  let html = "";
  for (const type of TYPE_ORDER) {
    const group = machines.filter((m) => m.type === type);
    if (!group.length) continue;
    html += `<h2>${TYPE_LABEL[type]}</h2><ul>`;
    for (const m of group) {
      let flag = "";
      if (m.status === "under_construction") flag = ` <small>*</small>`;
      else if (m.offline) flag = ` <small>&dagger;</small>`;
      html += `<li><a href="machine.html?m=${encodeURIComponent(m.name)}" target="main">${m.name}</a>${flag}</li>`;
    }
    html += `</ul>`;
  }
  html += `<p><small>* under construction<br>&dagger; offline for maintenance</small></p>`;
  el.innerHTML = html;
}

function renderMachine(machines, el) {
  const name = new URLSearchParams(location.search).get("m");
  const idx = machines.findIndex((m) => m.name === name);
  if (idx < 0) {
    el.innerHTML = `
    <h1>machine not found</h1>
    <p><a href="home.html">back to the floor</a>.</p>
    ${footerHTML()}`;
    return;
  }
  const m = machines[idx];
  const plate = "GGUF-" + String(idx + 1).padStart(3, "0");
  const offline = !!m.offline;
  const operational = m.status === "operational" && !offline;
  document.title = `${m.name} — after the bubble`;

  const statusCell = offline
    ? `<strong>OFFLINE</strong> <small>(temporarily, for maintenance)</small>`
    : operational
    ? `<strong>OPERATIONAL</strong>`
    : `<strong class="blink">UNDER CONSTRUCTION</strong>`;

  let html = ''

  if (m.explanation && m.explanation.length) {
    html += `<details><summary>What is this?</summary>` + m.explanation.join("\n") + `</details>`;
  }

  if (operational) {
    html += `
    <iframe src="../${m.links.demo}" width="100%" height="780"></iframe>`;
  } else if (offline) {
    html += `
    <p align="center"><strong>MACHINE OFFLINE.</strong><br>
    <small>check back later.</small></p>`;
  } else {
    html += `
    <p align="center"><img src="assets/construction.gif" width="232" height="56"
      alt="UNDER CONSTRUCTION"></p>
    <p align="center">this machine is not ready yet. maybe it will never be.</p>`;
  }

  html += `
    <h1>${m.name}</h1>
    <table border="1" cellpadding="4" width="100%">
      <caption>FACTORY NAMEPLATE</caption>
      <tr><th scope="row">plate no.</th><td>${plate}</td>
          <th scope="row">type</th><td>${TYPE_LABEL[m.type] || m.type}</td></tr>
      <tr><th scope="row">gguf size</th><td>${m.gguf_size}</td>
          <th scope="row">clock</th><td>${m.clock}</td></tr>
      <tr><th scope="row">status</th><td colspan="3">${statusCell}</td></tr>
      <tr><th scope="row">duty</th><td colspan="3">${m.blurb || ""}</td></tr>
    </table>`;

  const link = (label, href, blank) => href
    ? `[<a href="${href}"${blank ? ` target="_blank"` : ""}>${label}</a>]`
    : `[${label}]`;
  html += `
    <p><strong>
      ${link("DOWNLOAD .GGUF", m.links.gguf_download)}
      ${link("GENERATOR SCRIPT", m.links.generator_script, true)}
    </strong></p>`;

  html += footerHTML();
  el.innerHTML = html;
}
