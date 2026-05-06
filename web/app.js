const state = {
  scans: [],
  currentScanId: "",
  file: null,
  imageDataUrl: "",
  imageId: "",
  imageWidth: 0,
  imageHeight: 0,
  boxes: [],
  selectedId: "",
  zoom: 1,
  drag: null,
  faceSessionId: "",
  personNames: {},
};

const el = {
  fileInput: document.getElementById("fileInput"),
  detectButton: document.getElementById("detectButton"),
  detectAllButton: document.getElementById("detectAllButton"),
  cropButton: document.getElementById("cropButton"),
  exportAllButton: document.getElementById("exportAllButton"),
  autoOrientationInput: document.getElementById("autoOrientationInput"),
  addBoxButton: document.getElementById("addBoxButton"),
  deleteBoxButton: document.getElementById("deleteBoxButton"),
  rotateLeftButton: document.getElementById("rotateLeftButton"),
  rotateRightButton: document.getElementById("rotateRightButton"),
  thresholdInput: document.getElementById("thresholdInput"),
  thresholdValue: document.getElementById("thresholdValue"),
  areaInput: document.getElementById("areaInput"),
  areaValue: document.getElementById("areaValue"),
  paddingInput: document.getElementById("paddingInput"),
  paddingValue: document.getElementById("paddingValue"),
  faceThresholdInput: document.getElementById("faceThresholdInput"),
  faceThresholdValue: document.getElementById("faceThresholdValue"),
  faceMarginInput: document.getElementById("faceMarginInput"),
  faceMarginValue: document.getElementById("faceMarginValue"),
  groupFacesButton: document.getElementById("groupFacesButton"),
  saveDestInput: document.getElementById("saveDestInput"),
  pickFolderButton: document.getElementById("pickFolderButton"),
  topKInput: document.getElementById("topKInput"),
  topKValue: document.getElementById("topKValue"),
  minPhotosInput: document.getElementById("minPhotosInput"),
  minPhotosValue: document.getElementById("minPhotosValue"),
  saveAllButton: document.getElementById("saveAllButton"),
  saveFacesButton: document.getElementById("saveFacesButton"),
  scanImage: document.getElementById("scanImage"),
  scanList: document.getElementById("scanList"),
  overlay: document.getElementById("overlay"),
  canvasWrap: document.getElementById("canvasWrap"),
  statusText: document.getElementById("statusText"),
  scanTitle: document.getElementById("scanTitle"),
  resultsGridCurrent: document.getElementById("resultsGridCurrent"),
  resultsGridAll: document.getElementById("resultsGridAll"),
  faceGroups: document.getElementById("faceGroups"),
  exportPath: document.getElementById("exportPath"),
  zoomInButton: document.getElementById("zoomInButton"),
  zoomOutButton: document.getElementById("zoomOutButton"),
  zoomLabel: document.getElementById("zoomLabel"),
};

function setStatus(text) {
  el.statusText.textContent = text;
}

function selectedBox() {
  return state.boxes.find((box) => box.id === state.selectedId);
}

function currentScan() {
  return state.scans.find((scan) => scan.localId === state.currentScanId);
}

function saveCurrentScanState() {
  const scan = currentScan();
  if (!scan) return;
  scan.imageId = state.imageId;
  scan.boxes = state.boxes;
  scan.selectedId = state.selectedId;
  scan.results = scan.results || [];
}

function restoreScan(scan) {
  saveCurrentScanState();
  state.currentScanId = scan.localId;
  state.file = scan.file;
  state.imageDataUrl = scan.dataUrl;
  state.imageId = scan.imageId || "";
  state.imageWidth = scan.width;
  state.imageHeight = scan.height;
  state.boxes = scan.boxes || [];
  state.selectedId = scan.selectedId || state.boxes[0]?.id || "";
  el.scanImage.src = scan.dataUrl;
  el.canvasWrap.classList.remove("empty");
  el.scanTitle.textContent = scan.name;
  setStatus(`${scan.width} x ${scan.height}. ${state.boxes.length ? `${state.boxes.length} crops ready for review.` : "Ready to detect."}`);
  const fitZoom = Math.min(1, (window.innerWidth - 390) / scan.width, 700 / scan.height);
  setZoom(fitZoom || 1);
  renderScanList();
  renderCurrentAndAll();
}

function allSessionResults() {
  const results = [];
  state.scans.forEach((scan) => {
    (scan.results || []).forEach((r) => results.push(r));
  });
  return results;
}

function renderGrid(container, results) {
  container.innerHTML = "";
  results.forEach((result) => {
    const orientation = result.orientation?.label ? ` · ${result.orientation.label}` : "";
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <img src="${result.url}" alt="${result.filename}">
      <div>
        <strong>${result.filename}</strong>
        <span>${result.width} x ${result.height}${orientation}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderCurrentAndAll() {
  const scan = currentScan();
  renderGrid(el.resultsGridCurrent, scan?.results || []);
  renderGrid(el.resultsGridAll, allSessionResults());
  updateControls();
}

function renderScanList() {
  el.scanList.innerHTML = "";
  if (!state.scans.length) {
    const empty = document.createElement("div");
    empty.className = "scan-badge";
    empty.textContent = "No scans loaded";
    el.scanList.appendChild(empty);
    return;
  }
  state.scans.forEach((scan, index) => {
    const button = document.createElement("button");
    button.className = `scan-item ${scan.localId === state.currentScanId ? "active" : ""}`;
    button.innerHTML = `
      <img src="${scan.dataUrl}" alt="">
      <span>
        <strong>${index + 1}. ${scan.name}</strong>
        <span>${scan.width} x ${scan.height}</span>
      </span>
      <span class="scan-badge">${scan.boxes?.length || 0}</span>
    `;
    button.addEventListener("click", () => restoreScan(scan));
    el.scanList.appendChild(button);
  });
}

function updateControls() {
  const hasImage = Boolean(state.imageDataUrl);
  const hasSelected = Boolean(selectedBox());
  const hasBoxes = state.boxes.length > 0;
  const scan = currentScan();
  const hasResults = Boolean(scan?.results?.length);
  const hasAnyResults = state.scans.some((s) => (s.results || []).length);
  const hasFaceSession = Boolean(state.faceSessionId);
  el.detectButton.disabled = !hasImage;
  el.detectAllButton.disabled = !state.scans.length;
  el.cropButton.disabled = !hasImage || !hasBoxes;
  el.exportAllButton.disabled = !state.scans.some((scan) => scan.boxes?.length);
  el.addBoxButton.disabled = !hasImage;
  el.deleteBoxButton.disabled = !hasSelected;
  el.rotateLeftButton.disabled = !hasSelected;
  el.rotateRightButton.disabled = !hasSelected;
  el.zoomInButton.disabled = !hasImage;
  el.zoomOutButton.disabled = !hasImage;
  el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  el.groupFacesButton.disabled = !hasAnyResults;
  el.saveAllButton.disabled = !hasAnyResults;
  el.saveFacesButton.disabled = !hasFaceSession;
}

function updateSettingLabels() {
  el.thresholdValue.textContent = el.thresholdInput.value;
  el.areaValue.textContent = `${(Number(el.areaInput.value) / 10).toFixed(1)}%`;
  el.paddingValue.textContent = `${el.paddingInput.value}px`;
  el.faceThresholdValue.textContent = (Number(el.faceThresholdInput.value) / 100).toFixed(2);
  el.faceMarginValue.textContent = (Number(el.faceMarginInput.value) / 100).toFixed(2);
  el.topKValue.textContent = String(Number(el.topKInput.value));
  el.minPhotosValue.textContent = String(Number(el.minPhotosInput.value));
}

function screenPoint(event) {
  const rect = el.overlay.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / state.zoom,
    y: (event.clientY - rect.top) / state.zoom,
  };
}

function clampPoint(point) {
  return [
    Math.max(0, Math.min(state.imageWidth - 1, point[0])),
    Math.max(0, Math.min(state.imageHeight - 1, point[1])),
  ];
}

function pointsToAttr(points) {
  return points.map((point) => point.join(",")).join(" ");
}

function renderOverlay() {
  el.overlay.innerHTML = "";
  el.overlay.setAttribute("viewBox", `0 0 ${state.imageWidth} ${state.imageHeight}`);
  el.overlay.style.width = `${state.imageWidth * state.zoom}px`;
  el.overlay.style.height = `${state.imageHeight * state.zoom}px`;

  state.boxes.forEach((box, index) => {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", pointsToAttr(box.points));
    polygon.classList.add("crop-shape");
    if (box.id === state.selectedId) polygon.classList.add("selected");
    polygon.addEventListener("pointerdown", (event) => {
      state.selectedId = box.id;
      const start = screenPoint(event);
      state.drag = {
        type: "move",
        boxId: box.id,
        start,
        original: box.points.map((point) => [...point]),
      };
      polygon.setPointerCapture(event.pointerId);
      renderOverlay();
    });
    el.overlay.appendChild(polygon);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const angleStr = box.angle && Math.abs(box.angle) > 1 ? ` ·${Math.round(box.angle)}°` : "";
    label.textContent = `${index + 1}${angleStr}`;
    label.setAttribute("x", box.points[0][0] + 12);
    label.setAttribute("y", box.points[0][1] + 26);
    label.classList.add("crop-label");
    el.overlay.appendChild(label);

    if (box.id === state.selectedId) {
      box.points.forEach((point, pointIndex) => {
        const handle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        handle.setAttribute("cx", point[0]);
        handle.setAttribute("cy", point[1]);
        handle.setAttribute("r", 8);
        handle.classList.add("crop-handle");
        handle.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
          state.drag = { type: "point", boxId: box.id, pointIndex };
          handle.setPointerCapture(event.pointerId);
        });
        el.overlay.appendChild(handle);
      });
    }
  });

  updateControls();
}

function setZoom(nextZoom) {
  state.zoom = Math.max(0.15, Math.min(2.5, nextZoom));
  el.scanImage.style.width = `${state.imageWidth * state.zoom}px`;
  el.scanImage.style.height = `${state.imageHeight * state.zoom}px`;
  el.canvasWrap.style.width = `${state.imageWidth * state.zoom}px`;
  el.canvasWrap.style.height = `${state.imageHeight * state.zoom}px`;
  renderOverlay();
}

async function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function loadImageSize(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = reject;
    image.src = dataUrl;
  });
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "Request failed");
  return data;
}

async function handleFile(file) {
  const dataUrl = await readFileAsDataUrl(file);
  const size = await loadImageSize(dataUrl);
  const scan = {
    localId: crypto.randomUUID(),
    file,
    name: file.name,
    dataUrl,
    width: size.width,
    height: size.height,
    imageId: "",
    boxes: [],
    selectedId: "",
    results: [],
  };
  state.scans.push(scan);
  if (!state.currentScanId) restoreScan(scan);
  renderScanList();
  updateControls();
}

async function detectPhotos(scan = currentScan(), makeActive = true) {
  if (!scan) return;
  if (makeActive && scan.localId !== state.currentScanId) restoreScan(scan);
  setStatus(`Detecting photo regions in ${scan.name}...`);
  const data = await postJson("/api/detect", {
    image: scan.dataUrl,
    threshold: Number(el.thresholdInput.value),
    minAreaRatio: Number(el.areaInput.value) / 1000,
    padding: Number(el.paddingInput.value),
  });
  scan.imageId = data.imageId;
  scan.width = data.width;
  scan.height = data.height;
  scan.boxes = data.boxes.map((box) => ({ ...box, rotation: 0, angle: box.angle || 0 }));
  scan.selectedId = scan.boxes[0]?.id || "";
  if (scan.localId === state.currentScanId) {
    state.imageId = scan.imageId;
    state.imageWidth = scan.width;
    state.imageHeight = scan.height;
    state.boxes = scan.boxes;
    state.selectedId = scan.selectedId;
    renderOverlay();
  }
  renderScanList();
  const skewed = scan.boxes.filter((b) => Math.abs(b.angle || 0) > 2).length;
  const skewNote = skewed > 0 ? ` (${skewed} skewed)` : "";
  setStatus(`Detected ${scan.boxes.length} crop${scan.boxes.length === 1 ? "" : "s"} in ${scan.name}${skewNote}.`);
}

async function detectAllScans() {
  saveCurrentScanState();
  for (let i = 0; i < state.scans.length; i += 1) {
    await detectPhotos(state.scans[i], false);
    setStatus(`Detected ${i + 1} of ${state.scans.length} scans.`);
  }
  const active = currentScan();
  if (active) restoreScan(active);
  setStatus(`Finished detecting ${state.scans.length} scans.`);
}

function addCrop() {
  const width = state.imageWidth * 0.32;
  const height = state.imageHeight * 0.28;
  const x = (state.imageWidth - width) / 2;
  const y = (state.imageHeight - height) / 2;
  const box = {
    id: crypto.randomUUID(),
    points: [
      [x, y],
      [x + width, y],
      [x + width, y + height],
      [x, y + height],
    ].map(clampPoint),
    rotation: 0,
    angle: 0,
    status: "manual",
  };
  state.boxes.push(box);
  saveCurrentScanState();
  state.selectedId = box.id;
  setStatus("Manual crop added. Drag corners to fit the photo.");
  renderOverlay();
}

function deleteCrop() {
  state.boxes = state.boxes.filter((box) => box.id !== state.selectedId);
  state.selectedId = state.boxes[0]?.id || "";
  saveCurrentScanState();
  renderOverlay();
}

function rotateSelected(direction) {
  const box = selectedBox();
  if (!box) return;
  box.rotation = ((box.rotation || 0) + direction + 4) % 4;
  saveCurrentScanState();
  setStatus(`Selected crop will rotate ${box.rotation * 90} degrees on export.`);
  renderOverlay();
}

// NOTE: results are always shown as two stacked grids (current + all).

function renderFaceGroups(data) {
  el.faceGroups.innerHTML = "";
  if (!data?.persons?.length) {
    const empty = document.createElement("p");
    empty.className = "face-groups-empty";
    empty.textContent = data?.total_faces === 0 ? "No faces detected in these photos." : "No persons found.";
    el.faceGroups.appendChild(empty);
    return;
  }
  const header = document.createElement("div");
  header.className = "face-groups-head";
  header.innerHTML = `<h2>Face Groups</h2><span>${data.total_faces} face${data.total_faces === 1 ? "" : "s"} · ${data.persons.length} person${data.persons.length === 1 ? "" : "s"}</span>`;
  el.faceGroups.appendChild(header);

  data.persons.forEach((person, idx) => {
    const card = document.createElement("div");
    card.className = "person-card";
    const displayName = state.personNames[person.id] || `Person ${idx + 1}`;
    const facesHtml = person.faces.map((f) => `
      <div class="face-thumb">
        <img src="${f.url}" alt="${f.source}" title="${f.source} · score ${f.score}">
      </div>
    `).join("");
    card.innerHTML = `
      <div class="person-header">
        <span class="person-badge">P${idx + 1}</span>
        <input class="text-input person-name" data-person-id="${person.id}" value="${displayName}" />
        <span>${person.count} photo${person.count === 1 ? "" : "s"}</span>
      </div>
      <div class="person-faces">${facesHtml}</div>
    `;
    el.faceGroups.appendChild(card);
  });

  el.faceGroups.querySelectorAll("input[data-person-id]").forEach((input) => {
    input.addEventListener("input", (event) => {
      const id = event.target.getAttribute("data-person-id");
      state.personNames[id] = event.target.value;
    });
  });
}

function sessionExportIds() {
  return state.scans
    .filter((s) => s.imageId && (s.results || []).length)
    .map((s) => s.imageId);
}

async function saveAllCrops() {
  const exportIds = sessionExportIds();
  if (!exportIds.length) return;
  const destDir = (el.saveDestInput.value || "").trim();
  if (!destDir) {
    setStatus("Please set Destination folder first.");
    return;
  }
  setStatus("Saving all exported crops...");
  el.saveAllButton.disabled = true;
  try {
    const data = await postJson("/api/save_all", { exportIds, destDir });
    setStatus(`Saved ${data.saved} image${data.saved === 1 ? "" : "s"} to ${data.outDir}.`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  } finally {
    el.saveAllButton.disabled = false;
  }
}

async function saveFaceFolders() {
  if (!state.faceSessionId) return;
  const destDir = (el.saveDestInput.value || "").trim();
  if (!destDir) {
    setStatus("Please set Destination folder first.");
    return;
  }
  const topK = Number(el.topKInput.value);
  const minPhotos = Number(el.minPhotosInput.value);
  setStatus("Saving face-wise folders...");
  el.saveFacesButton.disabled = true;
  try {
    const data = await postJson("/api/save_faces", {
      sessionId: state.faceSessionId,
      destDir,
      topK,
      minPhotos,
      names: state.personNames,
    });
    setStatus(`Saved ${data.saved} photo${data.saved === 1 ? "" : "s"} into ${data.folders} folder${data.folders === 1 ? "" : "s"}: ${data.outDir}.`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  } finally {
    el.saveFacesButton.disabled = false;
  }
}

async function groupFaces() {
  const exportIds = state.scans
    .filter((s) => s.imageId && (s.results || []).length)
    .map((s) => s.imageId);
  if (!exportIds.length) return;
  setStatus("Detecting and grouping faces…");
  el.groupFacesButton.disabled = true;
  try {
    const data = await postJson("/api/faces_session", {
      exportIds,
      threshold: Number(el.faceThresholdInput.value) / 100,
      margin:    Number(el.faceMarginInput.value) / 100,
    });
    state.faceSessionId = data.sessionId || "";
    // reset default names for this run
    state.personNames = {};
    (data.persons || []).forEach((p, idx) => {
      state.personNames[p.id] = `Person ${idx + 1}`;
    });
    renderFaceGroups(data);
    const n = data.persons?.length || 0;
    setStatus(`Found ${n} person${n === 1 ? "" : "s"} across ${data.total_faces} face${data.total_faces === 1 ? "" : "s"}.`);
  } catch (err) {
    setStatus(`Face grouping failed: ${err.message}`);
  } finally {
    el.groupFacesButton.disabled = false;
    updateControls();
  }
}

async function generateResults(scan = currentScan(), makeActive = true) {
  if (!scan || !scan.boxes?.length) return null;
  saveCurrentScanState();
  if (makeActive && scan.localId !== state.currentScanId) restoreScan(scan);
  setStatus(`Generating cropped photos for ${scan.name}...`);
  const data = await postJson("/api/crop", {
    imageId: scan.imageId,
    image: scan.dataUrl,
    boxes: scan.boxes,
    autoOrientation: el.autoOrientationInput.checked,
  });
  scan.imageId = data.imageId || scan.imageId;
  scan.results = data.results;
  if (scan.localId === state.currentScanId) {
    renderCurrentAndAll();
    el.exportPath.textContent = data.exportPath;
  }
  setStatus(`Generated ${data.results.length} cropped photo${data.results.length === 1 ? "" : "s"}.`);
  return data;
}

async function exportAllReviewed() {
  saveCurrentScanState();
  const reviewed = state.scans.filter((scan) => scan.boxes?.length);
  for (let i = 0; i < reviewed.length; i += 1) {
    await generateResults(reviewed[i], false);
    setStatus(`Exported ${i + 1} of ${reviewed.length} reviewed scans.`);
  }
  const active = currentScan();
  if (active) restoreScan(active);
  setStatus(`Finished exporting ${reviewed.length} reviewed scans.`);
}

async function pickFolder() {
  setStatus("Opening folder picker...");
  try {
    const data = await postJson("/api/pick_folder", {});
    if (data.path) {
      el.saveDestInput.value = data.path;
      setStatus("Destination folder selected.");
    }
  } catch (err) {
    setStatus(err.message);
  }
}

el.fileInput.addEventListener("change", (event) => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  setStatus(`Loading ${files.length} scan${files.length === 1 ? "" : "s"}...`);
  Promise.all(files.map((file) => handleFile(file)))
    .then(() => setStatus(`Loaded ${state.scans.length} scan${state.scans.length === 1 ? "" : "s"}.`))
    .catch((error) => setStatus(error.message));
});

el.detectButton.addEventListener("click", () => detectPhotos().catch((error) => setStatus(error.message)));
el.detectAllButton.addEventListener("click", () => detectAllScans().catch((error) => setStatus(error.message)));
el.cropButton.addEventListener("click", () => generateResults().catch((error) => setStatus(error.message)));
el.exportAllButton.addEventListener("click", () => exportAllReviewed().catch((error) => setStatus(error.message)));
el.addBoxButton.addEventListener("click", addCrop);
el.deleteBoxButton.addEventListener("click", deleteCrop);
el.rotateLeftButton.addEventListener("click", () => rotateSelected(3));
el.rotateRightButton.addEventListener("click", () => rotateSelected(1));
el.zoomInButton.addEventListener("click", () => setZoom(state.zoom + 0.1));
el.zoomOutButton.addEventListener("click", () => setZoom(state.zoom - 0.1));

[el.thresholdInput, el.areaInput, el.paddingInput, el.faceThresholdInput, el.faceMarginInput, el.topKInput, el.minPhotosInput].forEach((input) => {
  input.addEventListener("input", updateSettingLabels);
});

el.groupFacesButton.addEventListener("click", () => groupFaces().catch((err) => setStatus(err.message)));
el.saveAllButton.addEventListener("click", () => saveAllCrops().catch((err) => setStatus(err.message)));
el.saveFacesButton.addEventListener("click", () => saveFaceFolders().catch((err) => setStatus(err.message)));
el.pickFolderButton.addEventListener("click", () => pickFolder());

window.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const box = state.boxes.find((item) => item.id === state.drag.boxId);
  if (!box) return;
  const point = screenPoint(event);
  if (state.drag.type === "point") {
    box.points[state.drag.pointIndex] = clampPoint([point.x, point.y]);
  } else if (state.drag.type === "move") {
    const dx = point.x - state.drag.start.x;
    const dy = point.y - state.drag.start.y;
    box.points = state.drag.original.map((original) => clampPoint([original[0] + dx, original[1] + dy]));
  }
  saveCurrentScanState();
  renderOverlay();
});

window.addEventListener("pointerup", () => {
  state.drag = null;
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Delete" || event.key === "Backspace") deleteCrop();
  if (event.key.toLowerCase() === "a") addCrop();
  if (event.key.toLowerCase() === "r") rotateSelected(1);
});

updateSettingLabels();
renderScanList();
updateControls();
